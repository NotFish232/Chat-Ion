import logging
import os
from typing import Callable

import torch as T
import torch.multiprocessing as mp
from torch import distributed as dist
from torch import nn, optim
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torchvision.transforms import Lambda
from tqdm import tqdm

from models import Network
from train.arg_parser import get_args
from utils import *
from utils.datasets import CornellMovieDataset


def setup_distributed(rank: int = -1, world_size: int = -1) -> None:
    dist.init_process_group(
        "nccl",
        rank=rank,
        init_method="tcp://localhost:8080",
        world_size=world_size,
    )


def cleanup_distributed() -> None:
    dist.destroy_process_group()


def is_main_process(rank: int, world_size: int) -> bool:
    return (rank == -1 and world_size == -1) or (rank == 0 and world_size > 0)


def is_multi_gpu(rank: int, world_size: int) -> bool:
    return rank != -1 and world_size != -1


def prepare_dataloader(
    batch_size: int,
    max_seq_len: int,
    rank: int,
    world_size: int,
    transforms: Callable = None,
    target_transforms: Callable = None,
) -> DataLoader:
    max_sentence_len = max_seq_len // 4
    max_passage_len = max_seq_len
    dataset = CornellMovieDataset(
        max_context_length=max_passage_len,
        max_sentence_length=max_sentence_len,
        transforms=transforms,
        target_transforms=target_transforms,
    )
    sampler = (
        InterleavedSampler(len(dataset), rank, world_size)
        if is_multi_gpu(rank, world_size)
        else None
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    return dataloader


def prepare_network(
    model_kwargs: dict[str, int | float],
    rank: int,
    world_size: int,
    device: T.device,
) -> nn.Module:
    network = Network(**model_kwargs).to(device)
    if is_multi_gpu(rank, world_size):
        setup_distributed(rank, world_size)
        network = DDP(network, device_ids=[rank])
    network.train()
    return network


def prepare_optimizer(
    network: nn.Module, learning_rate: float, weight_decay: float
) -> optim.Optimizer:
    # layer norm and biases shouldn't be weight decayed
    decay_params = []
    no_decay_params = []
    for param in network.parameters():
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)
    optimizer_params = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0},
    ]

    optimizer = optim.AdamW(
        optimizer_params,
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-6,
    )

    return optimizer


def prepare_logger(rank: int, world_size: int) -> logging.Logger:
    logger = logging.getLogger(__name__)

    level = logging.INFO if is_main_process(rank, world_size) else logging.CRITICAL
    logger.setLevel(level)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    logger.addHandler(ch)

    return logger


def training_loop(
    model_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    max_seq_len: int,
    checkpoint_interval: int,
    device: str,
    model_kwargs: dict,
    rank: int = -1,
    world_size: int = -1,
) -> None:
    logger = prepare_logger(rank, world_size)
    device = T.device(f"cuda:{rank}" if is_multi_gpu(rank, world_size) else device)
    vocab = Vocabulary()

    transforms = Lambda(lambda x: T.tensor(x, device=device))
    dataloader = prepare_dataloader(
        batch_size,
        max_seq_len,
        rank=rank,
        world_size=world_size,
        transforms=transforms,
        target_transforms=transforms,
    )

    network = prepare_network(
        model_kwargs | {"num_embed": len(vocab)},
        rank,
        world_size,
        device=device,
    )

    logger.info(f"Parameters: {sum(i.numel() for i in network.parameters()):,}")

    optimizer = prepare_optimizer(network, learning_rate, weight_decay)

    scheduler = CosineAnnealingWarmRestarts(optimizer, 20)

    scaler = amp.GradScaler() if device.type == "cuda" else None

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.PAD_IDX)

    model_mgr = ModelManager(model_name)

    if model_mgr.checkpoint_exists():
        logger.info("Checkpoint exists, loading save!")
        checkpoint = model_mgr.load_checkpoint(network, optimizer, scheduler, scaler)
        (starting_epochs,) = checkpoint
    else:
        logger.info("Checkpoint doesn't exist, creating new model.")
        starting_epochs = 0

    batch_idx = 0
    for epoch in range(starting_epochs + 1, starting_epochs + epochs + 1):
        num_correct = 0
        num_total = 0
        total_loss = 0

        for prompts, labels in tqdm(
            dataloader,
            desc=f"Epoch {epoch}",
            disable=not is_main_process(rank, world_size),
        ):
            labels_input = labels[:, :-1]
            labels_expected = labels[:, 1:]

            masks = {
                "tgt_mask": make_look_ahead_mask(labels_input.size(-1), device),
                "src_key_padding_mask": prompts == vocab.PAD_IDX,
                "tgt_key_padding_mask": labels_input == vocab.PAD_IDX,
            }

            with amp.autocast(enabled=device.type == "cuda"):
                y = network(prompts, labels_input, **masks)

                loss = criterion(
                    y.view(-1, len(vocab)),
                    labels_expected.flatten(),
                )

            total_loss += loss.item()

            if device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            optimizer.zero_grad()
            scheduler.step()

            num_correct += T.sum(T.argmax(y, dim=-1) == labels_expected).item()
            num_total += prompts.size(0) * prompts.size(1)

            if (
                batch_idx != 0
                and batch_idx % checkpoint_interval == 0
                and is_main_process(rank, world_size)
            ):
                logger.info("Saving checkpoint...")
                accuracy = num_correct / num_total
                model_mgr.save_checkpoint(network, optimizer, scheduler, scaler, epoch)

            batch_idx += 1

        avg_loss = total_loss / len(dataloader)

        accuracy = num_correct / num_total
        logger.info(f"Accuracy: {accuracy:.2%}, loss: {avg_loss:.2f}")

    if is_main_process(rank, world_size):
        logger.info("Making final checkpoint and saving model...")
        model_mgr.save_checkpoint(network, optimizer, scheduler, scaler, epoch)
        model_mgr.save_model(network, model_kwargs)

    cleanup_distributed()


def _training_loop_helper(rank: int, world_size: int, kwargs: dict) -> None:
    training_loop(rank=rank, world_size=world_size, **kwargs)


def main() -> None:
    args = get_args()
    training_args = args["training"]
    model_args = args["model"]

    # both a training and a model arg
    training_args["max_seq_len"] = model_args["max_seq_len"]

    print(f"Training with arguments: {training_args | model_args}")

    num_gpus = training_args.pop("num_gpus")

    if training_args["device"] == "cuda":
        world_size = num_gpus if num_gpus != -1 else T.cuda.device_count()
        os.environ["OPENBLAS_NUM_THREADS"] = "20"
        mp.spawn(
            _training_loop_helper,
            args=(world_size, training_args | {"model_kwargs": model_args}),
            nprocs=world_size,
            join=True,
        )
    else:
        training_loop(**training_args, model_kwargs=model_args)


if __name__ == "__main__":
    main()
