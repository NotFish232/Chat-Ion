from models.network import Network
from utils.checkpointer import CheckPointer
from utils.dataset import ConversationDataset

import torch as T
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision.transforms import Lambda
from tqdm import tqdm

BATCH_SIZE = 256
NUM_EPOCHS = 150
EMBED_DIM = 512
LEARNING_RATE = 1e-5


def main() -> None:
    device = T.device("cuda" if T.cuda.is_available() else "cpu")

    transforms = Lambda(lambda x: T.tensor(x, device=device))

    dataset = ConversationDataset(transforms=transforms, target_transforms=transforms)
    dataloader = DataLoader(dataset, BATCH_SIZE)

    network = Network(dataset.num_words, EMBED_DIM).to(device)
    print(f"Parameters: {sum(i.numel() for i in network.parameters()):,}")

    optimizer = optim.Adam(network.parameters(), LEARNING_RATE)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.PAD_IDX)

    checkpointer = CheckPointer()

    if checkpointer.checkpoint_exists():
        print("Checkpoint exists, loading save!")
        x = checkpointer.load_checkpoint()
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

    src_look_ahead_mask = T.triu(
        T.ones(
            dataset.max_sentence_length,
            dataset.max_sentence_length,
            dtype=T.bool,
            device=device,
        ),
        diagonal=1,
    )
    tgt_look_ahead_mask = T.triu(
        T.ones(
            dataset.max_sentence_length + 1,
            dataset.max_sentence_length + 1,
            dtype=T.bool,
            device=device,
        ),
        diagonal=1,
    )

    for epoch in range(starting_epochs + 1, starting_epochs + NUM_EPOCHS + 1):
        num_correct = 0
        num_total = 0
        total_loss = 0

        for prompts, labels in tqdm(dataloader, desc=f"Epoch {epoch}"):
            labels_input = labels[:, :-1]
            labels_expected = labels[:, 1:]

            masks = {
                "src_mask": src_look_ahead_mask,
                "tgt_mask": tgt_look_ahead_mask,
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
        print(f"Accuracy: {accuracy:.2%}")

    if accuracy > starting_accuracy:
        print(f"Accuracy improved! ({accuracy:.2%} vs {starting_accuracy:.2%})")
        print("Saving checkpoint...")
        checkpointer.save(network, optimizer, scheduler, epoch, accuracy)
    else:
        print(f"Accuracy decreased. ({accuracy:.2%} vs {starting_accuracy:.2%})")
        print("Not saving checkpoint.")


if __name__ == "__main__":
    main()
