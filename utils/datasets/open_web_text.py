import io
import json
import multiprocessing as mp
from enum import Enum
from typing import Callable, Iterator
import itertools

import random
import nltk
from .vocabulary import Vocabulary
import zstandard as zstd
from torch.utils.data import Dataset
from tqdm import tqdm
from typing_extensions import Self

from .shared import DATA_DIR

"""
modes for training
pass in sentence -> model outputs sentence
pass in sentence -> model outputs rest of passage
pass in most of passage -> model outputs last sentence
pass in half the passage -> model outputs other half of passage
masks random tokens, like in BERT
"""


class Modes(Enum):
    SentToSent = 0
    SentToPass = 1
    PassToSent = 2
    PassToPass = 3
    Masking = 4


PASSAGE_END = "### PASSAGE END ###"

MASKING_PROB = 0.15
RANDOM_WORD_PROB = 0.1
ACTUAL_WORD_PROB = 0.1


class OpenWebTextDataset(Dataset):
    def __init__(
        self: Self,
        mode: Modes | str,
        folder_name: str = "openwebtext2",
        unprocessed_file_dir: str = "unprocessed",
        processed_file_name: str = "processed.zst",
        info_file_name: str = "info.json",
        max_sentence_length: int = 64,
        max_passage_length: int = 256,
        transforms: Callable = None,
        target_transforms: Callable = None,
    ) -> None:
        assert isinstance(mode, (Modes, str)), "mode must be of type Modes or str"
        if isinstance(mode, str):
            assert (
                mode in Modes.__members__
            ), f"mode '{mode} not avaliable, possible modes are {[k for k in Modes.__members__]}"
            mode = Modes.__members__.get(mode)
        self.mode = mode

        self.data_dir = DATA_DIR / folder_name
        self.unprocessed_file_dir = unprocessed_file_dir
        self.processed_file_name = processed_file_name
        self.info_file_name = info_file_name

        self.max_sentence_length = max_sentence_length
        self.max_passage_length = max_passage_length

        self.transforms = transforms
        self.target_transforms = target_transforms

        self.vocab = Vocabulary()

        self.files = list((self.data_dir / unprocessed_file_dir).glob("*.jsonl.zst"))

        if not (self.data_dir / processed_file_name).exists():
            self._process_data()

        self.num_passages, self.num_sentences = self._read_info_file()

        self.current_idx = 0

        # necessry bc apparently you cant prev an iter
        if self.mode == Modes.SentToSent:
            self.previous_sentence = None

        sentence_iterator = self._read_jsonl(self.data_dir / processed_file_name)
        self.sentence_gen = itertools.cycle(sentence_iterator)

    def __len__(self: Self) -> int:
        return (
            self.num_sentences - self.num_passages
            if self.mode == Modes.SentToSent
            else self.num_passages
        )

    """
    supports arbitrary indexing, but in reality if you are using 
    this you should definitely only retrieve sequentially
    otherwise it's going to take way to long to get to an
    arbitrary index in the dataset
    """

    def __getitem__(self: Self, idx: int) -> tuple:
        if self.mode == Modes.SentToSent and self.previous_sentence is None:
            self.previous_sentence = next(self.sentence_gen)

        while self.current_idx != idx:
            line = next(self.sentence_gen)

            if self.mode == Modes.SentToSent:
                if line != PASSAGE_END:
                    self.current_idx += 1
                    self.previous_sentence = line
                else:
                    self.previous_sentence = None
            else:
                if line == PASSAGE_END:
                    self.current_idx += 1

        if self.mode == Modes.Masking:
            passage = ""

            while True:
                line = next(self.sentence_gen)
                if line == PASSAGE_END:
                    break
                passage += line

            tokens = self.vocab.tokenize(passage)
            input = []
            label = []
            for token in tokens:
                if random.random() <= MASKING_PROB:
                    prob = random.random()

                    if prob < ACTUAL_WORD_PROB:
                        input.append(token)
                    elif prob < ACTUAL_WORD_PROB + RANDOM_WORD_PROB:
                        input.append(random.randint(0, self.vocab.num_reg_tokens - 1))
                    else:
                        input.append(self.vocab.MASK_IDX)
                    label.append(token)
                else:
                    input.append(token)
                    label.append(self.vocab.PAD_IDX)

            return input, label

    @staticmethod
    def _read_jsonl(file_path: str) -> Iterator[str]:
        zstd_ctx = zstd.ZstdDecompressor()
        with open(file_path, "rb") as f:
            reader = io.BufferedReader(zstd_ctx.stream_reader(f))
            for line in reader:
                yield json.loads(line)

    def _tokenize_passage(self: Self, passage: str) -> list[str]:
        return nltk.sent_tokenize(passage)

    def _process_file(self: Self, file: str) -> dict:
        sentences = []
        num_passages = 0
        num_sentences = 0

        for passage in self._read_jsonl(file):
            new_sentences = self._tokenize_passage(passage["text"])
            sentences.extend(new_sentences)
            sentences.append(PASSAGE_END)
            num_sentences += len(new_sentences)
            num_passages += 1

        return {
            "sentences": sentences,
            "num_passages": num_passages,
            "num_sentences": num_sentences,
        }

    def _process_data(self: Self) -> dict:
        num_passages = 0
        num_sentences = 0

        num_processes = 64
        compressor = zstd.ZstdCompressor()

        with (
            mp.Pool(num_processes) as p,
            open(self.data_dir / self.processed_file_name, "wb+") as f,
            compressor.stream_writer(f) as compressed_writer,
        ):
            m = p.imap(self._process_file, self.files)

            for r in tqdm(m, total=len(self.files)):
                for sentence in r["sentences"]:
                    compressed_writer.write(json.dumps(sentence).encode() + b"\n")

                num_passages += r["num_passages"]
                num_sentences += r["num_sentences"]

        info_json = {
            "num_passages": num_passages,
            "num_sentences": num_sentences,
        }

        with open(self.data_dir / self.info_file_name, "w+") as f:
            json.dump(info_json, f)

    def _read_info_file(self: Self) -> tuple:
        with open(self.data_dir / self.info_file_name, "r") as f:
            data = json.load(f)

        num_passages = data["num_passages"]
        num_sentences = data["num_sentences"]

        return num_passages, num_sentences


if __name__ == "__main__":
    c = OpenWebTextDataset()
