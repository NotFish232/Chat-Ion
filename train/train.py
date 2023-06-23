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
NUM_EPOCHS = 2
EMBED_DIM = 512
LEARNING_RATE = 1e-4


def main() -> None:
    device = T.device("cuda" if T.cuda.is_available() else "cpu")

    transforms = Lambda(lambda x: T.tensor(x, device=device))

    dataset = ConversationDataset(transforms=transforms, target_transforms=transforms)
    dataloader = DataLoader(dataset, BATCH_SIZE)

    network = Network(dataset.num_words, EMBED_DIM).to(device)
    print(f"Parameters: {sum(i.numel() for i in network.parameters()):,}")

    optimizer = optim.Adam(network.parameters(), LEARNING_RATE)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
    criterion = nn.CrossEntropyLoss()

    checkpointer = CheckPointer()

    if checkpointer.exists():
        print("Checkpoint exists, loading save!")
        x = checkpointer.load()
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

    eye = T.eye(dataset.num_words, device=device)

    look_ahead_mask = T.triu(
        T.ones(
            dataset.max_sentence_length,
            dataset.max_sentence_length,
            dtype=T.bool,
            device=device,
        ),
        diagonal=1,
    )

    for epoch in range(starting_epochs + 1, starting_epochs + NUM_EPOCHS + 1):
        num_correct = 0
        num_total = 0
        for prompts, labels in tqdm(dataloader, desc=f"Epoch {epoch}"):
            labels_input = labels[:, :-1]
            labels_expected = labels[:, 1:]

            y = network(prompts, labels_input, look_ahead_mask, look_ahead_mask)
            loss = criterion(y, eye[labels_expected])

            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            optimizer.zero_grad()

            num_correct += T.sum(T.argmax(y, dim=-1) == labels_expected).item()
            num_total += prompts.size(0) * dataset.max_sentence_length

        accuracy = num_correct / num_total
        print(f"Accuracy: {accuracy:.2%}")

    if accuracy > starting_accuracy:
        print(f"Accuracy improved! ({accuracy:.2%} vs {starting_accuracy:.2%})")
        print("Saving model...")
        checkpointer.save(network, optimizer, scheduler, epoch, accuracy)
    else:
        print(f"Accuracy decreased. ({accuracy:.2%} vs {starting_accuracy:.2%})")
        print("Not saving model.")


if __name__ == "__main__":
    main()
