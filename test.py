import torch as T
from network import Network
from dataset import ConversationDataset

EMBED_DIM = 256


def main():
    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    dataset = ConversationDataset()

    network = Network(dataset.num_words, EMBED_DIM).to(device)
    network.load_state_dict(T.load("trained_model.pt"))

    memory_mask = T.triu(
        T.ones(
            dataset.max_sentence_length,
            dataset.max_sentence_length,
            dtype=T.bool,
            device=device,
        ),
        diagonal=1,
    )

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
                y = network(sentence, tgt, memory_mask)
                response = T.argmax(y[0, t - 1])
                tgt[t] = response

            response = " ".join(map(lambda x: dataset.rvocab[x.item()], tgt))
            print(response)


if __name__ == "__main__":
    main()
