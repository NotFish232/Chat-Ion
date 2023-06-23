import torch as T
from network import Network
from dataset import ConversationDataset

EMBED_DIM = 256


def main():
    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    dataset = ConversationDataset()
    network = Network(dataset.num_words, EMBED_DIM).to(device)
    network.load_state_dict(T.load("trained_model.pt"))

    user_input = ""
    network.eval()
    with T.no_grad():
        while user_input != "quit":
            user_input = input(">> ")
            sentence = dataset.tokenize_sentence(user_input.split(" "))
            sentence = T.tensor(sentence, device=device)
            tgt = T.zeros(dataset.max_sentence_length, device=device, dtype=T.int32)
            tgt[0] = dataset.SOS_IDX
            for t in range(1, len(tgt)):
                y = T.argmax(network(sentence, tgt), dim=-1)[0]
                tgt[t] = y[t - 1]
            response = " ".join(map(lambda x: dataset.rvocab[x.item()], tgt))
            print(response)


if __name__ == "__main__":
    main()
