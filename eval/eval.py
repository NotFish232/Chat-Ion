import warnings

import torch as T

from models import Network
from utils import Vocabulary, ModelManager
from .arg_parser import get_args
from utils import make_look_ahead_mask


def run_evaluation(model_name: str, device: str) -> None:
    device = T.device(device)

    model_mgr = ModelManager(model_name)

    vocab = Vocabulary()

    model_kwargs = model_mgr.load_model_info()
    network = Network(len(vocab), **model_kwargs).to(device)
    model_mgr.load_model(network)


    max_seq_len = model_kwargs["max_seq_len"]

    look_ahead_mask = make_look_ahead_mask(max_seq_len, device)

    user_input = ""
    network.eval()
    with T.no_grad():
        while user_input != "goodbye":
            user_input = input(">> ").lower()
            sentence = vocab.tokenize(sentence)
            sentence.extend(vocab.PAD_IDX for _ in range(max_seq_len - len(sentence)))
            sentence = T.tensor(sentence, device=device)

            tgt = T.full((max_seq_len + 1), vocab.PAD_IDX, device=device)
            tgt[0] = vocab.SOS_IDX

            for t in range(1, len(tgt)):
                masks = {
                    "tgt_mask": look_ahead_mask,
                    "src_key_padding_mask": sentence == vocab.PAD_IDX,
                    "tgt_key_padding_mask": tgt == vocab.PAD_IDX,
                }
                y = network(sentence, tgt, **masks)

                response = T.argmax(y[0, t - 1])
                tgt[t] = response

                if response == vocab.EOS_IDX:
                    break

                print(vocab.idx_to_token[response.item()], end=" ")

            print()


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)

    args = get_args()

    run_evaluation(**args)


if __name__ == "__main__":
    main()
