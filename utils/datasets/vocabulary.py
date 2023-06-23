import json
import string

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

        self.SOS_IDX = self.token_to_idx["<sos>"]
        self.EOS_IDX = self.token_to_idx["<eos>"]
        self.MASK_IDX = self.token_to_idx["<mask>"]
        self.OOV_IDX = self.token_to_idx["<oov>"]
        self.PAD_IDX = self.token_to_idx["<pad>"]
        self.CLS_IDX = self.token_to_idx["<cls>"]
        self.SEP_IDX = self.token_to_idx["<sep>"]

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

    def tokenize(self: Self, sentence: str, to_idxs: bool = True) -> list[str | int]:
        tokens = self.tokenizer.tokenize(sentence)
        if to_idxs:
            tokens = [self.token_to_idx.get(t, self.OOV_IDX) for t in tokens]
        return tokens

    def __len__(self: Self) -> int:
        return len(self.token_to_idx)

    def __getitem__(self: Self, word_or_idx: str | int) -> int | str:
        assert isinstance(
            word_or_idx, (str, int)
        ), "Index passed must be of type str or int"

        if isinstance(word_or_idx, str):
            return self.token_to_idx[word_or_idx]

        if isinstance(word_or_idx, int):
            return self.idx_to_token[word_or_idx]


if __name__ == "__main__":
    v = Vocabulary()
