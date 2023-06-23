import torch as T
from network import Network
from dataset import ConversationDataset
from torchvision.transforms import Compose, Lambda

EMBED_DIM = 256


def main():
    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    dataset = ConversationDataset()
    network = Network(dataset.num_words, EMBED_DIM).to(device)
    network.load_state_dict(T.load("trained_model.pt"))

    user_input = ""
    network.eval()
    while user_input != "quit":
        user_input = input(">> ")
        words = user_input.split(" ")
        sentence = T.tensor(dataset.tokenize_sentence(words), device=device)
        with T.no_grad():
            y = T.argmax(
                network(sentence, T.zeros(dataset.max_sentence_length, device=device, dtype=T.int32)),
                dim=1,
            )
        response = "".join(map(lambda x: dataset.rvocab[x], y[0]))
        print(response)


if __name__ == "__main__":
    main()
