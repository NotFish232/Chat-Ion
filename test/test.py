import torch as T
from models.network import Network
from utils.dataset import CornellMovieDataset
from utils.checkpointer import CheckPointer
import warnings

EMBED_DIM = 512


def main():
    warnings.filterwarnings("ignore", category=UserWarning)

    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    dataset = CornellMovieDataset()

    network = Network(dataset.num_words, EMBED_DIM).to(device)

    checkpointer = CheckPointer()

    network.load_state_dict(checkpointer.load_model())

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

    user_input = ""
    network.eval()
    with T.no_grad():
        while user_input != "goodbye":
            user_input = input(">> ").lower()
            sentence = dataset.tokenize_sentence(user_input.split(" "))
            sentence = T.tensor(sentence, device=device)

            tgt = T.full(
                (dataset.max_sentence_length + 1,), dataset.PAD_IDX, device=device
            )
            tgt[0] = dataset.SOS_IDX

            for t in range(1, len(tgt)):
                masks = {
                    "src_mask": src_look_ahead_mask,
                    "tgt_mask": tgt_look_ahead_mask,
                    "src_key_padding_mask": sentence == dataset.PAD_IDX,
                    "tgt_key_padding_mask": tgt == dataset.PAD_IDX,
                }
                y = network(sentence, tgt, **masks)

                response = T.argmax(y[0, t - 1])
                tgt[t] = response

                if response == dataset.EOS_IDX:
                    break

                print(dataset.rvocab[response.item()], end=" ")

            print()


if __name__ == "__main__":
    main()
