import torch as T
from torch import amp, nn, optim, distributed as dist
from torch.cuda.amp import GradScaler
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import Lambda
import torch.multiprocessing as mp
from tqdm import tqdm

from models import Network
from train.arg_parser import get_args
from utils import make_look_ahead_mask
from utils.datasets import CornellMovieDataset, Vocabulary
from utils.model_loader import ModelLoader


def setup_dist(rank: int, world_size: int) -> None:
    dist.init_process_group(
        "gloo",
        rank=rank,
        init_method="tcp://localhost:8080",
        world_size=world_size,
    )


def training_loop(
    rank: int,
    world_size: int,
    model_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    checkpoint_interval: int,
    device: str,
    model_kwargs: dict,
) -> None:
    if rank != 0:
        import os
        import sys

        f = open(os.devnull, "w")
        sys.stdout = f
    device = T.device(f"cuda:{rank}")

    transforms = Lambda(lambda x: T.tensor(x, device=device))

    vocab = Vocabulary()
    dataset = CornellMovieDataset(transforms=transforms, target_transforms=transforms)
    sampler = DistributedSampler(dataset, world_size, rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    network = Network(len(vocab), **model_kwargs)
    if device.type == "cuda":
        setup_dist(rank, world_size)
        network = DDP(network.to(device), device_ids=[rank])

    print(f"Parameters: {sum(i.numel() for i in network.parameters()):,}")

    optimizer = optim.Adam(
        network.parameters(), learning_rate, weight_decay=weight_decay
    )
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
    scaler = GradScaler()

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.PAD_IDX)

    model_loader = ModelLoader()

    if model_loader.checkpoint_exists():
        print("Checkpoint exists, loading save!")
        x = model_loader.load_checkpoint()

        network.load_state_dict(x["network"])
        optimizer.load_state_dict(x["optimizer"])
        scheduler.load_state_dict(x["scheduler"])
        scaler.load_state_dict(x["scaler"])
        starting_epochs = x["epochs"]
        starting_accuracy = x["accuracy"]

        print(f"Loaded model ({starting_accuracy:.2%})")
    else:
        print("Checkpoint doesn't exist, creating new model.")
        starting_epochs = 0
        starting_accuracy = 0

    for epoch in range(starting_epochs + 1, starting_epochs + epochs + 1):
        num_correct = 0
        num_total = 0
        total_loss = 0

        for prompts, labels in tqdm(dataloader, desc=f"Epoch {epoch}"):
            labels_input = labels[:, :-1]
            labels_expected = labels[:, 1:]

            masks = {
                "tgt_mask": make_look_ahead_mask(labels_input.size(-1), device),
                "src_key_padding_mask": prompts == vocab.PAD_IDX,
                "tgt_key_padding_mask": labels_input == vocab.PAD_IDX,
            }

            with amp.autocast(device.type):
                y = network(prompts, labels_input, **masks)

                loss = criterion(
                    y.view(-1, len(vocab)),
                    labels_expected.flatten(),
                )

            total_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()

            num_correct += T.sum(T.argmax(y, dim=-1) == labels_expected).item()
            num_total += prompts.size(0) * dataset.max_sentence_length

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)

        accuracy = num_correct / num_total
        print(f"Accuracy: {accuracy:.2%}, loss: {avg_loss:.2f}")

        if epoch % checkpoint_interval == 0:
            print("Saving checkpoint...")
            model_loader.save_checkpoint(
                network, optimizer, scaler, scheduler, epoch, accuracy
            )

    model_loader.save_model(network)


def _training_helper(rank: int, world_size: int, kwargs: dict) -> None:
    training_loop(rank, world_size, **kwargs)


def main() -> None:
    args = get_args()

    print(f"Training with arguments: {args['training'] | args['model']}")

    world_size = T.cuda.device_count()

    mp.spawn(
        _training_helper,
        args=(world_size, args["training"] | {"model_kwargs": args["model"]}),
        nprocs=world_size,
        join=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
