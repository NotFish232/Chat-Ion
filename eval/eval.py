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
    network.load_state_dict(model_mgr.load_model())

    max_seq_len = model_kwargs["max_seq_len"]

    look_ahead_mask = make_look_ahead_mask(max_seq_len, device)

    user_input = ""
    network.eval()
    with T.no_grad():
        while user_input.lower() != "quit":
            user_input = input(">> ")
            sentence_parts = user_input.split(" ? ")  # '?' should be predicted by model
            tokens = []
            for i, part in enumerate(sentence_parts):
                if i != 0:
                    tokens.append(vocab.MASK_IDX)
                new_tokens = vocab.tokenize(part)
                tokens.extend(new_tokens)
            tokens = vocab.fix_length(tokens, max_seq_len, add_cls_and_sep=True)
            masked_idxs = [i for i, t in enumerate(tokens) if t == vocab.MASK_IDX]
            sentence = T.tensor(tokens, device=device).unsqueeze(0)

            tgt = T.full((1, max_seq_len), vocab.PAD_IDX, device=device)

            for t in range(tgt.size(1)):
                if t not in masked_idxs:
                    continue

                masks = {
                    "tgt_mask": look_ahead_mask,
                    "src_key_padding_mask": sentence == vocab.PAD_IDX,
                    "tgt_key_padding_mask": tgt == vocab.PAD_IDX,
                }
                y = network(sentence, tgt, **masks)

                response = T.argmax(y[0, t])
                tgt[0, t] = response

            output = ""
            for i, part in enumerate(sentence_parts):
                output += part
                if i != len(sentence_parts) - 1:
                    output += f" {vocab[tgt[0, masked_idxs[i]].item()]} "
            print(output)


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)

    args = get_args()

    run_evaluation(**args)


if __name__ == "__main__":
    main()
