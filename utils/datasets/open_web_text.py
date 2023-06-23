import io
import json
import multiprocessing as mp
from typing import Callable, Iterator
from .shared import DATA_DIR
from enum import Enum


import nltk
import zstandard as zstd
from torch.utils.data import Dataset
from tqdm import tqdm
from typing_extensions import Self

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


class OpenWebTextDataset(Dataset):
    def __init__(
        self: Self,
        mode: Modes | str,
        folder_name: str = "openwebtext2",
        unprocessed_file_dir: str = "unprocessed",
        processed_file_name: str = "processed.zst",
        info_file_name: str = "info.json",
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

        self.transforms = transforms
        self.target_transforms = target_transforms

        self.files = list((self.data_dir / unprocessed_file_dir).glob("*.jsonl.zst"))

        if not (self.data_dir / processed_file_name).exists():
            self._process_data()

        self.num_passages, self.num_sentences = self._read_info_file()

        self.current_idx = 0

        # necessry bc apparently you cant prev an iter
        if self.mode == Modes.SentToSent:
            self.previous_sentence = None

        self.sentence_gen = self._read_jsonl(self.data_dir / processed_file_name)

    def __len__(self: Self) -> int:
        return (
            self.num_sentences - self.num_passages
            if self.mode == Modes.SentToSent
            else self.num_sentences
            if self.mode == Modes.Masking
            else self.num_passages
        )

    """
    supports arbitrary indexing, but in reality if you are using 
    this you should definitely only retrieve sequentially
    otherwise it's going to take way to long to get to an
    arbitrary index in the dataset
    """
    def __getitem__(self: Self, idx: int) -> tuple:
        if self.mode == "sentence_to_sentence" and self.previous_sentence is None:
            self.previous_sentence = json.loads(next(self.sentence_gen))

        while self.current_idx != idx:
            line = json.loads(next(self.sentence_gen))

            if self.mode == "sentence_to_sentence":
                if line != PASSAGE_END:
                    self.current_idx += 1
                else:
                    self.previous_sentence = None
            else:
                if line == PASSAGE_END:
                    self.current_idx += 1

    @staticmethod
    def _read_jsonl(file_path: str) -> Iterator[str]:
        zstd_ctx = zstd.ZstdDecompressor()
        with open(file_path, "rb") as f:
            reader = io.BufferedReader(zstd_ctx.stream_reader(f))
            for line in reader:
                json_line = json.loads(line)
                yield json_line["text"]

    def _tokenize_sentence(self: Self, sentence: str) -> list[str]:
        return nltk.word_tokenize(sentence)

    def _tokenize_passage(self: Self, passage: str) -> list[str]:
        return nltk.sent_tokenize(passage)

    def _process_file(self: Self, file: str) -> dict:
        sentences = []
        num_passages = 0

        for passage in self._read_jsonl(file):
            new_sentences = self._tokenize_passage(passage)
            sentences.extend(new_sentences)
            num_passages += 1

        return {
            "sentences": sentences,
            "num_passages": num_passages,
        }

    def _process_data(self: Self) -> dict:
        num_processes = 32

        num_sentences = 0
        num_passages = 0

        compressor = zstd.ZstdCompressor()

        with (
            mp.Pool(num_processes) as p,
            open(self.data_dir / self.processed_file_name, "wb+") as f,
            compressor.stream_writer(f) as compressed_writer,
        ):
            m = p.imap(self._process_file, self.files)

            for r in tqdm(m, total=len(self.files)):
                for sentence in r["sentences"]:
                    compressed_writer.write(
                        json.dumps(sentence).encode("utf-8") + b"\n"
                    )
                    num_sentences += 1

                compressed_writer.write(json.dumps(PASSAGE_END).encode("utf-8") + b"\n")

                num_passages += r["num_passages"]

        info_json = {
            "num_sentences": num_sentences,
            "num_passages": num_passages,
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
