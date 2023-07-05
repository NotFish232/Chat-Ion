import json
import string

import nltk
from nltk import corpus
from transformers import BertTokenizer
from typing_extensions import Self

from .shared import DATA_DIR

SPECIAL_TOKENS = ["<sos>", "<eos>", "<mask>", "<oov>", "<pad>", "<cls>", "<sep>"]


class Vocabulary:
    def __init__(
        self: Self,
        folder_name: str = "vocabulary",
        vocab_file_name: str = "tokens.json",
    ) -> None:
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-cased", local_files_only=True
        )

        vocab_file_path = DATA_DIR / folder_name / vocab_file_name
        if not vocab_file_path.exists():
            # supplementary word list for more comprehensive vocab
            with open(DATA_DIR / folder_name / "en_comprehensive.txt", "r") as f:
                en_comprehensive = f.read().split("\n")

            words = (
                en_comprehensive
                + list(corpus.words.words())
                + list(corpus.wordnet.words())
            )
            lower_words = [w.lower() for w in words]
            capital_words = [w.capitalize() for w in words]
            upper_words = [w.upper() for w in words]
            all_words = lower_words + capital_words + upper_words

            self.tokens = list(set(self.tokenizer.tokenize(" ".join(all_words))))
            for punct in string.punctuation:
                if punct in self.tokens:
                    self.tokens.remove(punct)
            self.tokens += list(string.punctuation)
            self.tokens += SPECIAL_TOKENS

            with open(vocab_file_path, "w+") as f:
                json.dump(self.tokens, f)
        else:
            with open(vocab_file_path, "r") as f:
                self.tokens = json.load(f)

        self.token_to_idx = dict(zip(self.tokens, range(self.num_tokens)))
        self.idx_to_token = {v: k for k, v in self.token_to_idx.items()}

        self.SOS_TOKEN = "<sos>"
        self.EOS_TOKEN = "<eos>"
        self.MASK_TOKEN = "<mask>"
        self.OOV_TOKEN = "<oov>"
        self.PAD_TOKEN = "<pad>"
        self.CLS_TOKEN = "<cls>"
        self.SEP_TOKEN = "<sep>"

        self.SOS_IDX = self.token_to_idx[self.SOS_TOKEN]
        self.EOS_IDX = self.token_to_idx[self.EOS_TOKEN]
        self.MASK_IDX = self.token_to_idx[self.MASK_TOKEN]
        self.OOV_IDX = self.token_to_idx[self.OOV_TOKEN]
        self.PAD_IDX = self.token_to_idx[self.PAD_TOKEN]
        self.CLS_IDX = self.token_to_idx[self.CLS_TOKEN]
        self.SEP_IDX = self.token_to_idx[self.SEP_TOKEN]

    @property
    def num_tokens(self: Self) -> int:
        return len(self.tokens)

    @property
    def num_reg_tokens(self: Self) -> int:
        return self.num_tokens - len(SPECIAL_TOKENS)

    def __new__(self: Self) -> "Vocabulary":
        if not hasattr(self, "instance"):
            self.instance = super(Vocabulary, self).__new__(self)
        return self.instance

    def fix_length(
        self: Self,
        tokens: list[int],
        n: int,
        add_sos_and_eos: bool = False,
        add_cls_and_sep: bool = False,
        truncate_from_left: bool = False,
    ) -> list[int]:
        if n == -1:
            return tokens

        if add_sos_and_eos or add_cls_and_sep:
            max_length = n - 2
        else:
            max_length = n

        if len(tokens) > max_length:
            tokens = tokens[-max_length:] if truncate_from_left else tokens[:max_length]

        if add_sos_and_eos:
            tokens.insert(0, self.SOS_IDX)
            tokens.append(self.EOS_IDX)

        if add_cls_and_sep:
            tokens.insert(0, self.CLS_IDX)
            tokens.append(self.SEP_IDX)

        tokens.extend(self.PAD_IDX for _ in range(n - len(tokens)))

        return tokens

    """
    tokenizes a sentence or list of sentences,
    converts to indexes, and fixes length to n
    """

    def tokenize(
        self: Self,
        sentence: str | list[str],
        n: int = -1,
        add_sos_and_eos: bool = False,
        add_cls_and_sep: bool = False,
        truncate_from_left: bool = False,
    ) -> list[int]:
        assert not (
            add_sos_and_eos and add_cls_and_sep
        ), "Only one of add_sos_and_eos and add_cls_and_sep should be provided"
        if isinstance(sentence, list):
            assert (
                add_cls_and_sep
            ), "add_cls_and_sep must be true if sentence provided is a list of sentences"
            tokens = []
            for i, sent in enumerate(sentence):
                tokens.extend(self.tokenizer.tokenize(sent))
                if i != len(sent) - 1:
                    tokens.append(self.SEP_TOKEN)
        else:
            tokens = self.tokenizer.tokenize(sentence)

        tokens = [self.token_to_idx.get(t, self.OOV_IDX) for t in tokens]

        tokens = self.fix_length(
            tokens,
            n,
            add_sos_and_eos,
            add_cls_and_sep,
            truncate_from_left,
        )

        return tokens

    def __len__(self: Self) -> int:
        return len(self.token_to_idx)

    def __getitem__(self: Self, word_or_idx: str | int) -> int | str:
        assert isinstance(
            word_or_idx, (str, int)
        ), "Index passed must be of type str or int"

        if isinstance(word_or_idx, str):
            return self.token_to_idx.get(word_or_idx, self.OOV_IDX)

        if isinstance(word_or_idx, int):
            return self.idx_to_token.get(word_or_idx, self.OOV_TOKEN)


if __name__ == "__main__":
    v = Vocabulary()
