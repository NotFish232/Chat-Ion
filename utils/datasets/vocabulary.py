import json
import string
from nltk import corpus
from transformers import BertTokenizer
from typing_extensions import Self
from .shared import DATA_DIR

SPECIAL_TOKENS = ["<sos>", "<eos>", "<mask>", "<oov>", "<pad>", "<cls>", "<sep>"]

CORPUS_DICT = {"words": corpus.words, "brown": corpus.brown}


class Vocabulary:
    def __init__(
        self: Self, corpus_name: str = "words", folder_name: str = "vocabulary"
    ) -> None:
        assert (
            corpus_name in CORPUS_DICT
        ), f"Corpus {corpus_name} not found, avaliable corpuses are {CORPUS_DICT.keys()}"

        corpus = CORPUS_DICT[corpus_name]

        lower_words = set(w.lower() for w in corpus.words())
        capital_words = set(w.capitalize() for w in lower_words)
        upper_words = set(w.upper() for w in lower_words)
        words = lower_words.union(capital_words).union(upper_words)

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

        vocab_file_path = DATA_DIR / folder_name / "tokens.json"
        if not vocab_file_path.exists():
            tokens = list(set(self.tokenizer.tokenize(" ".join(words))))
            tokens += list(string.punctuation)
            tokens += SPECIAL_TOKENS

            with open(vocab_file_path, "w+") as f:
                json.dump(tokens, f)
        else:
            with open(vocab_file_path, "r") as f:
                tokens = json.load(f)

        self.token_to_idx = dict(zip(tokens, range(len(tokens))))
        self.idx_to_token = {v: k for k, v in self.token_to_idx.items()}

        self.SOS_IDX = self.token_to_idx["<sos>"]
        self.EOS_IDX = self.token_to_idx["<eos>"]
        self.MASK_IDX = self.token_to_idx["<mask>"]
        self.OOV_IDX = self.token_to_idx["<oov>"]
        self.PAD_IDX = self.token_to_idx["<pad>"]
        self.CLS_IDX = self.token_to_idx["<cls>"]
        self.SEP_IDX = self.token_to_idx["<sep>"]

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
