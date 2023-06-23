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
        unprocessed_file_dir: str = "unprocessed",
        processed_folder_name: str = "processed",
        num_processed_files: int = 6,
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
        self.processed_folder_name = processed_folder_name
        self.num_processed_files = num_processed_files
        self.info_file_name = info_file_name

        self.max_sentence_length = max_sentence_length
        self.max_passage_length = max_passage_length

        self.transforms = transforms
        self.target_transforms = target_transforms

        self.vocab = Vocabulary()

        self.files = list((self.data_dir / unprocessed_file_dir).glob("*.jsonl.zst"))

        if not (self.data_dir / self.info_file_name).exists():
            self._process_data()

        self.num_passages, self.num_sentences = self._read_info_file()

        self.current_idx = 0

        sentence_iterator = self._read_jsonl(self.data_dir / processed_file_name)
        self.sentence_gen = itertools.cycle(sentence_iterator)

        # necessry bc apparently you cant prev an iter
        if self.mode == Modes.SentToSent:
            self.previous_sentence = next(self.sentence_gen)

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

            if self.current_idx == len(self):
                self.current_idx = 0

        input, target = (
            self._make_sent_to_sent_task()
            if self.mode == Modes.SentToSent
            else self._make_sent_to_pass_task()
            if self.mode == Modes.SentToPass
            else self._make_pass_to_sent_task()
            if self.mode == Modes.PassToSent
            else self._make_pass_to_pass_task()
            if self.mode == Modes.PassToPass
            else self._make_masking_task()
            if self.mode == Modes.Masking
            else (None, None)
        )

        self.current_idx += 1
        if self.current_idx == len(self):
            self.current_idx = 0

        if self.transforms is not None:
            input = self.transforms(input)
        if self.target_transforms is not None:
            target = self.target_transforms(target)

        return input, target

    def _make_sent_to_sent_task(self: Self) -> tuple[list[int], list[int]]:
        input = self.previous_sentence
        target = next(self.sentence_gen)
        if target == PASSAGE_END:
            input = next(self.sentence_gen)
            target = next(self.sentence_gen)
        self.previous_sentence = target

        input = self.vocab.tokenize(input)
        target = self.vocab.tokenize(target)

        input = self._pad_tokens(input, self.max_sentence_length, True)
        target = self._pad_tokens(
            target, self.max_sentence_length, add_sos_and_eos=True
        )

        return input, target

    def _make_sent_to_pass_task(self: Self) -> tuple[list[int], list[int]]:
        input = next(self.sentence_gen)
        target = ""

        while True:
            line = next(self.sentence_gen)
            if line == PASSAGE_END:
                break
            target += line

        input = self.vocab.tokenize(input)
        target = self.vocab.tokenize(target)

        input = self._pad_tokens(input, self.max_sentence_length, True)
        target = self._pad_tokens(target, self.max_passage_length, add_sos_and_eos=True)

        return input, target

    def _make_pass_to_sent_task(self: Self) -> tuple[list[int], list[int]]:
        sentences = []

        while True:
            line = next(self.sentence_gen)
            if line == PASSAGE_END:
                break
            sentences.append(line)

        input = "".join(sentences[:-1])
        target = sentences[-1]

        input = self.vocab.tokenize(input)
        target = self.vocab.tokenize(target)

        input = self._pad_tokens(input, self.max_passage_length, True)
        target = self._pad_tokens(
            target, self.max_sentence_length, add_sos_and_eos=True
        )

        return input, target

    def _make_pass_to_pass_task(self: Self) -> tuple[list[int], list[int]]:
        sentences = []

        while True:
            line = next(self.sentence_gen)
            if line == PASSAGE_END:
                break
            sentences.append(line)

        middle = len(sentences) // 2 + 1

        input = "".join(sentences[:middle])
        target = "".join(sentences[middle:])

        input = self.vocab.tokenize(input)
        target = self.vocab.tokenize(target)

        input = self._pad_tokens(input, self.max_passage_length, True)
        target = self._pad_tokens(target, self.max_passage_length, add_sos_and_eos=True)

        return input, target

    def _make_masking_task(self: Self) -> tuple[list[int], list[int]]:
        passage = ""
        while True:
            line = next(self.sentence_gen)
            if line == PASSAGE_END:
                break
            passage += line

        tokens = self.vocab.tokenize(passage)
        tokens = self._pad_tokens(tokens, self.max_passage_length, add_sos_and_eos=True)
        input, target = [], []
        for token in tokens:
            if random.random() <= MASKING_PROB:
                prob = random.random()
                if prob < ACTUAL_WORD_PROB:
                    input.append(token)
                elif prob < ACTUAL_WORD_PROB + RANDOM_WORD_PROB:
                    input.append(random.randint(0, self.vocab.num_reg_tokens - 1))
                else:
                    input.append(self.vocab.MASK_IDX)
                target.append(token)
            else:
                input.append(token)
                target.append(self.vocab.PAD_IDX)

        return input, target

    def _pad_tokens(
        self: Self,
        tokens: list[int],
        n: int,
        trim_left: bool = False,
        add_sos_and_eos: bool = False,
    ) -> list[int]:
        if add_sos_and_eos:
            n -= 1
        if len(tokens) > n:
            tokens = tokens[-n:] if trim_left else tokens[:n]

        num_pads = n - len(tokens)

        if add_sos_and_eos:
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
            m = p.imap(self._process_file, self.files)

            for r in tqdm(m, total=len(self.files), desc="reading unprocessed..."):
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

        num_passages = data["num_passages"]
        num_sentences = data["num_sentences"]

        return num_passages, num_sentences


if __name__ == "__main__":
    c = OpenWebTextDataset()
