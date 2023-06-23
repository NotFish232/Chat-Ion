from network import Network
from dataset import ConversationDataset
import torch as T
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda
from tqdm import tqdm

BATCH_SIZE = 256
NUM_EPOCHS = 10
EMBED_DIM = 256
LEARNING_RATE = 1e-3


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
    dataloader = DataLoader(dataset, BATCH_SIZE, drop_last=True)

    network = Network(dataset.num_words, EMBED_DIM).to(device)
    print(f"{sum(i.numel() for i in network.parameters()):,}")

    optimizer = optim.Adam(network.parameters(), LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    eye = T.eye(dataset.num_words, device=device)
    
    for epoch in range(1, NUM_EPOCHS + 1):
        acc_loss = 0
        for prompt, labels in tqdm(dataloader, desc=f"Epoch {epoch}"):
            y = network(prompt, labels)
            loss = criterion(y, eye[labels])
            acc_loss += loss.detach()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Loss {acc_loss:.2f}")
    T.save(network.state_dict(), "trained_model.pt")


if __name__ == "__main__":
    main()
