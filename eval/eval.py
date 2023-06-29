import warnings

import torch as T

from models import Transformer
from utils import ModelManager, Vocabulary, make_look_ahead_mask

from .arg_parser import get_args


def run_evaluation(model_name: str, device: str) -> None:
    device = T.device(device)

    model_mgr = ModelManager(model_name)

    vocab = Vocabulary()

    model_kwargs = model_mgr.load_model_info()
    network = Transformer(len(vocab), **model_kwargs).to(device)
    model_mgr.load_model(network)

    max_seq_len = model_kwargs["max_seq_len"]

    look_ahead_mask = make_look_ahead_mask(max_seq_len, device)

    user_input = ""
    network.eval()
    with T.no_grad():
        while user_input != "goodbye":
            user_input = input(">> ").lower()
            sentence = [vocab.CLS_IDX] + vocab.tokenize(user_input)
            sentence.extend(vocab.PAD_IDX for _ in range(max_seq_len - len(sentence)))
            sentence = T.tensor(sentence, device=device).unsqueeze(0)

            tgt = T.full((1, max_seq_len + 1), vocab.PAD_IDX, device=device)
            tgt[0, 0] = vocab.SOS_IDX
            print(tgt.shape)

            for t in range(1, tgt.size(-1)):
                tgt_input = tgt[:, :-1]
                masks = {
                    "tgt_mask": look_ahead_mask,
                    "src_key_padding_mask": sentence == vocab.PAD_IDX,
                    "tgt_key_padding_mask": tgt_input == vocab.PAD_IDX,
                }
                y = network(sentence, tgt_input, **masks)

                response = T.argmax(y[0, t - 1])
                tgt[0, t] = response

                if response.item() == vocab.EOS_IDX:
                    break

                print(vocab.idx_to_token[response.item()], end=" ")

            print()


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)

    args = get_args()

    run_evaluation(**args)


if __name__ == "__main__":
    main()
