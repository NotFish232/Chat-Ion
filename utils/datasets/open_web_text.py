import io
import itertools
import json
import multiprocessing as mp
import random
from enum import Enum
from typing import Callable, Iterator

import nltk
import zstandard as zstd
from torch.utils.data import Dataset
from tqdm import tqdm
from typing_extensions import Self

from .shared import DATA_DIR
from .vocabulary import Vocabulary

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
        unprocessed_folder_name: str = "unprocessed",
        processed_folder_name: str = "processed",
        num_processed_files: int = 128,
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
        self.unprocessed_folder_name = unprocessed_folder_name
        self.processed_folder_name = processed_folder_name
        self.num_processed_files = num_processed_files
        self.info_file_name = info_file_name

        self.max_sentence_length = max_sentence_length
        self.max_passage_length = max_passage_length

        self.transforms = transforms
        self.target_transforms = target_transforms

        self.vocab = Vocabulary()

        self.unprocessed_files = list(
            (self.data_dir / unprocessed_folder_name).glob("*.jsonl.zst")
        )

        if not (self.data_dir / self.info_file_name).exists():
            self._process_data()

        self.processed_files = list(
            (self.data_dir / processed_folder_name).glob("*.zst")
        )

        self.num_passages, self.num_sentences = self._read_info_file()

        self.current_idxs = [0 for _ in range(num_processed_files)]

        self.sentence_generators = [
            itertools.cycle(self._read_jsonl(f)) for f in self.processed_files
        ]

    def __len__(self: Self) -> int:
        return self.num_passages

    """
    supports arbitrary indexing, but in reality if you are using 
    this you should definitely only retrieve sequentially
    otherwise it's going to take way to long to get to an
    arbitrary index in the dataset
    """

    def __getitem__(self: Self, _idx: int) -> tuple:
        bucket_idx = _idx % self.num_processed_files
        sentence_generator = self.sentence_generators[bucket_idx]
        idx = _idx // self.num_processed_files

        while self.current_idxs[bucket_idx] != idx:
            line = next(sentence_generator)

            if line == PASSAGE_END:
                self.current_idxs[bucket_idx] += 1

            if self.current_idxs[bucket_idx] == self.__len__():
                self.current_idxs[bucket_idx] = 0

        passage = self._get_passage(sentence_generator)
        self.current_idxs[bucket_idx] += 1
        if self.current_idxs[bucket_idx] == self.__len__():
            self.current_idxs[bucket_idx] = 0

        input, target = (
            (passage[0], passage[1])
            if self.mode == Modes.SentToSent
            else (passage[0], passage[1:])
            if self.mode == Modes.SentToPass
            else (passage[:-1], passage[-1])
            if self.mode == Modes.PassToSent
            else (passage[: len(passage) // 2], passage[len(passage) // 2 :])
            if self.mode == Modes.PassToPass
            else (passage, passage)
            if Modes.Masking
            else (None, None)
        )

        input = self.vocab.tokenize(input)
        target = self.vocab.tokenize(target)

        input = self._pad_tokens(input, False)
        target = self._pad_tokens(target)

        if self.mode == Modes.Masking:
            input, target = self._mask(input, target)

        if self.transforms is not None:
            input = self.transforms(input)
        if self.transforms is not None:
            target = self.target_transforms(target)

        return input, target

    def _mask(
        self: Self, input: list[str], target: list[str]
    ) -> tuple[list[int], list[int]]:
        for idx in range(len(input)):
            if input[idx] == self.vocab.PAD_IDX:
                break

            if random.random() <= MASKING_PROB:
                prob = random.random()
                if prob < RANDOM_WORD_PROB:
                    input[idx] = random.randint(0, self.vocab.num_reg_tokens - 1)
                elif prob < 1 - ACTUAL_WORD_PROB:
                    input[idx] = self.vocab.MASK_IDX
            else:
                target[idx] = self.vocab.PAD_IDX

        return input, target

    def _get_passage(self: Self, sentence_generator: Iterator) -> list[str]:
        passage = []
        while True:
            line = next(sentence_generator)
            if line == PASSAGE_END:
                break
            passage.append(line)
        return passage

    def _pad_tokens(self: Self, tokens: list[int], is_target: bool = True) -> list[int]:
        n = (
            self.max_sentence_length
            if self.mode == Modes.SentToSent
            or (not is_target and self.mode == Modes.PassToSent)
            or (is_target and self.mode == Modes.SentToPass)
            else self.max_sentence_length
        )

        if is_target:
            n -= 1

        if len(tokens) > n:
            tokens = tokens[-n:] if not is_target else tokens[:n]

        num_pads = n - len(tokens)

        if is_target:
            tokens.insert(0, self.vocab.SOS_IDX)
            tokens.append(self.vocab.EOS_IDX)

        tokens.extend(self.vocab.PAD_IDX for _ in range(num_pads))

        return tokens

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
        passages = []

        for passage in self._read_jsonl(file):
            sentences = self._tokenize_passage(passage["text"])
            sentences.append(PASSAGE_END)
            passages.append(sentences)

        return passages

    def _process_data(self: Self) -> dict:
        passages = []

        num_processes = 64
        compressor = zstd.ZstdCompressor()

        with mp.Pool(num_processes) as p:
            m = p.imap(self._process_file, self.unprocessed_files)

            for r in tqdm(
                m, total=len(self.unprocessed_files), desc="reading unprocessed..."
            ):
                passages.extend(r)

        num = len(passages) // self.num_processed_files

        info_json = []

        for n in tqdm(range(self.num_processed_files), desc="writing processed..."):
            with (
                open(
                    self.data_dir / self.processed_folder_name / f"{n}.zst", "wb+"
                ) as f,
                compressor.stream_writer(f) as compressed_writer,
            ):
                num_passages = 0
                num_sentences = 0
                for idx in range(n * num, (n + 1) * num):
                    for sentence in passages[idx]:
                        compressed_writer.write(json.dumps(sentence).encode() + b"\n")
                        num_sentences += 1
                    num_passages += 1

                info_json.append(
                    {"num_passages": num_passages, "num_sentences": num_sentences}
                )

        with open(self.data_dir / self.info_file_name, "w+") as f:
            json.dump(info_json, f)

    def _read_info_file(self: Self) -> tuple:
        with open(self.data_dir / self.info_file_name, "r") as f:
            data = json.load(f)

        total_passages = 0
        total_sentences = 0

        for d in data:
            total_passages += d["num_passages"]
            total_sentences += d["num_sentences"]

        return total_passages, total_sentences


if __name__ == "__main__":
    c = OpenWebTextDataset()
