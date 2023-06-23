from argparse import Namespace

import torch as T
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.transforms import Lambda
from tqdm import tqdm

from models import Network
from train.arg_parser import get_args
from utils.datasets import CornellMovieDataset
from utils.model_loader import ModelLoader


def training_loop(args: Namespace) -> None:
    device = T.device(args.device)

    transforms = Lambda(lambda x: T.tensor(x, device=device))

    dataset = CornellMovieDataset(transforms=transforms, target_transforms=transforms)
    dataloader = DataLoader(dataset, args.batch_size)

    network = Network(dataset.num_words, args.embed_dim).to(device)
    print(f"Parameters: {sum(i.numel() for i in network.parameters()):,}")

    optimizer = optim.Adam(
        network.parameters(), args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.PAD_IDX)

    model_loader = ModelLoader()

    if model_loader.checkpoint_exists():
        print("Checkpoint exists, loading save!")
        x = model_loader.load_checkpoint()
        network.load_state_dict(x["network"])
        optimizer.load_state_dict(x["optimizer"])
        scheduler.load_state_dict(x["scheduler"])
        starting_epochs = x["epochs"]
        starting_accuracy = x["accuracy"]
        print(f"Loaded model ({starting_accuracy:.2%})")
    else:
        print("Checkpoint doesn't exist, creating new model.")
        starting_epochs = 0
        starting_accuracy = 0

    look_ahead_mask = T.triu(
        T.ones(
            dataset.max_sentence_length + 1,
            dataset.max_sentence_length + 1,
            dtype=T.bool,
            device=device,
        ),
        diagonal=1,
    )

    for epoch in range(starting_epochs + 1, starting_epochs + args.epochs + 1):
        num_correct = 0
        num_total = 0
        total_loss = 0

        for prompts, labels in tqdm(dataloader, desc=f"Epoch {epoch}"):
            labels_input = labels[:, :-1]
            labels_expected = labels[:, 1:]

            masks = {
                "tgt_mask": look_ahead_mask,
                "src_key_padding_mask": prompts == dataset.PAD_IDX,
                "tgt_key_padding_mask": labels_input == dataset.PAD_IDX,
            }

            y = network(prompts, labels_input, **masks)

            loss = criterion(
                y.view(-1, dataset.num_words),
                labels_expected.flatten(),
            )

            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            num_correct += T.sum(T.argmax(y, dim=-1) == labels_expected).item()
            num_total += prompts.size(0) * dataset.max_sentence_length

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)

        accuracy = num_correct / num_total
        print(f"Accuracy: {accuracy:.2%}, loss: {avg_loss:.2f}")

        if epoch % args.checkpoint == 0:
            print("Saving checkpoint...")
            model_loader.save_checkpoint(network, optimizer, scheduler, epoch, accuracy)

    model_loader.save_model(network)


def main() -> None:
    args = get_args()

    training_loop(args)


if __name__ == "__main__":
    main()
