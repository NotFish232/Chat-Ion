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

from models import Transformer
from train.arg_parser import get_args
from utils import *
from utils.datasets.shared import Modes
from utils.datasets import CornellMovieDataset, OpenWebTextDataset


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
    return not is_multi_gpu(rank, world_size) or rank == 0


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
    def custom_collate_fn(batch: list[dict]) -> dict:
        batch_dict = {}

        for data in batch:
            for key, val in data.items():
                if isinstance(val, (list, tuple, T.Tensor)):
                    if key not in batch_dict:
                        batch_dict[key] = []
                    batch_dict[key].append(val)
                else:
                    batch_dict[key] = val

        for key, val in batch[0].items():
            if isinstance(val, T.Tensor):
                batch_dict[key] = T.stack(batch_dict[key])

        return batch_dict

    max_sentence_len = max_seq_len // 4
    max_passage_len = max_seq_len

    dataset1 = CornellMovieDataset(
        max_sentence_length=max_sentence_len,
        max_context_length=max_passage_len,
        transforms=transforms,
        target_transforms=target_transforms,
    )
    dataset2 = OpenWebTextDataset(
        Modes.Masking,
        max_sentence_length=max_sentence_len,
        max_passage_length=max_passage_len,
        transforms=transforms,
        target_transforms=target_transforms,
    )

    if is_multi_gpu(rank, world_size):
        sampler1 = InterleavedSampler(len(dataset1), rank, world_size)
        sampler2 = InterleavedSampler(len(dataset2), rank, world_size)
    else:
        sampler1 = None
        sampler2 = None

    dataloader1 = DataLoader(
        dataset1, batch_size, sampler=sampler1, collate_fn=custom_collate_fn
    )
    dataloader2 = DataLoader(
        dataset2, batch_size, sampler=sampler2, collate_fn=custom_collate_fn
    )

    dataloader = InterleavedDataLoader(dataloader1, dataloader2)

    return dataloader


def prepare_network(
    model_kwargs: dict[str, int | float],
    rank: int,
    world_size: int,
    device: T.device,
) -> nn.Module:
    network = Transformer(**model_kwargs).to(device)
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
        if not param.requires_grad:
            continue

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


def calc_accuracy(y_hat: T.Tensor, y: T.Tensor, ignore_idx: int) -> int:
    mask = y != ignore_idx
    y_hat = y_hat[mask]
    y = y[mask]
    num_correct = T.sum(y_hat == y).item()
    num_total = y.numel()

    return num_correct / num_total


def training_loop(
    model_name: str,
    epochs: int,
    batch_size: int,
    grad_acc_steps: int,
    learning_rate: float,
    weight_decay: float,
    max_seq_len: int,
    checkpoint_interval: int,
    device: str,
    model_kwargs: dict = {},
    rank: int = -1,
    world_size: int = -1,
) -> None:
    logger = prepare_logger(rank, world_size)
    is_main = is_main_process(rank, world_size)

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
    scheduler = CosineAnnealingWarmRestarts(optimizer, 50)
    scaler = amp.GradScaler(enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.PAD_IDX)
    model_mgr = ModelManager(model_name)

    if model_mgr.checkpoint_exists():
        logger.info("Checkpoint exists, loading save!")
        checkpoint = model_mgr.load_checkpoint()
        network.load_state_dict(checkpoint["network"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        starting_epochs = checkpoint["epochs"]
    else:
        logger.info("Checkpoint doesn't exist, creating new model.")
        starting_epochs = 0

    iteration = 0
    for epoch in range(starting_epochs + 1, starting_epochs + epochs + 1):
        for batch in tqdm(
            dataloader,
            desc=f"Epoch {epoch}",
            disable=not is_main,
        ):
            source = batch["src"]
            target = batch["tgt"]
            mode = batch["mode"]
            # no need to shift output by 1 when running a masking task
            if mode == Modes.Masking:
                target_input = target_expected = target
            else:
                target_input = target[:, :-1]
                target_expected = target[:, 1:]

            masks = {
                "tgt_mask": make_look_ahead_mask(target_input.size(-1), device),
                "src_key_padding_mask": source == vocab.PAD_IDX,
                "tgt_key_padding_mask": target_input == vocab.PAD_IDX,
            }

            with amp.autocast(enabled=device.type == "cuda"):
                y = network(source, target_input, **masks)

                loss = criterion(
                    y.view(-1, len(vocab)),
                    target_expected.flatten(),
                )

            loss /= grad_acc_steps
            scaler.scale(loss).backward()

            iteration += 1

            if iteration % grad_acc_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            if iteration % checkpoint_interval == 0 and is_main:
                target_pred = T.argmax(y, dim=-1)
                accuracy = calc_accuracy(target_pred, target_expected, vocab.PAD_IDX)
                logger.info("Saving checkpoint...")
                logger.info(f"Accuracy: {accuracy:.2%}")
                checkpoint = {
                    "network": network.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epochs": epoch,
                }
                model_mgr.save_checkpoint(checkpoint)

    # step optimizer with final gradients
    if iteration % grad_acc_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

    if is_main:
        logger.info("Making final checkpoint and saving model...")
        checkpoint = {
            "network": network.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epochs": epoch,
        }
        model_mgr.save_checkpoint(checkpoint)
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
