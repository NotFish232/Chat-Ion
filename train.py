from network import Network
from dataset import ConversationDataset
import torch as T
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda
from tqdm import tqdm

BATCH_SIZE = 256
NUM_EPOCHS = 50
EMBED_DIM = 256
LEARNING_RATE = 1e-4


def main() -> None:
    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    transforms = Compose(
        [
            Lambda(lambda x: T.tensor(x, device=device)),
        ]
    )
    target_transforms = Compose(
        [
            Lambda(lambda x: T.tensor(x, device=device)),
        ]
    )
    dataset = ConversationDataset(
        transforms=transforms, target_transforms=target_transforms
    )
    dataloader = DataLoader(dataset, BATCH_SIZE)

    network = Network(dataset.num_words, EMBED_DIM).to(device)
    print(f"Parameters: {sum(i.numel() for i in network.parameters()):,}")

    optimizer = optim.Adam(network.parameters(), LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    eye = T.eye(dataset.num_words, device=device)

    memory_mask = T.triu(
        T.ones(
            dataset.max_sentence_length,
            dataset.max_sentence_length,
            dtype=T.bool,
            device=device,
        ),
        diagonal=1,
    )

    for epoch in range(1, NUM_EPOCHS + 1):
        num_correct = 0
        num_total = 0
        for prompt, labels in tqdm(dataloader, desc=f"Epoch {epoch}"):
            labels_input = labels[:, :-1]
            labels_expected = labels[:, 1:]

            y = network(prompt, labels_input, memory_mask)
            loss = criterion(y, eye[labels_expected])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            num_correct += T.sum(T.argmax(y, dim=-1) == labels_expected)
            num_total += prompt.size(0) * dataset.max_sentence_length

        print(f"Accuracy: {num_correct / num_total:.2%}")

    T.save(network.state_dict(), "trained_model.pt")


if __name__ == "__main__":
    main()
