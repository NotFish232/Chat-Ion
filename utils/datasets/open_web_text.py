import io
import json
from pathlib import Path
from typing import Callable, Iterator
from tqdm import tqdm
import multiprocessing as mp
from collections import Counter
import nltk

import zstandard as zstd
from torch.utils.data import Dataset
from typing_extensions import Self

BASE_DIR = Path(__file__).parents[2]

POSSIBLE_MODES = [
    "sentence_to_sentence",
    "sentence_to_paragraph",
    "paragraph_to_sentence",
    "paragraph_to_paragraph",
]

SPECIAL_TOKENS = ["<eos>", "<sos>", "<pad>", "<mask>", "<oov>"]

PASSAGE_END = "### PASSAGE END ###"


class OpenWebTextDataset(Dataset):
    def __init__(
        self: Self,
        mode: str,
        data_dir: str = "data/openwebtext2",
        transforms: Callable = None,
        target_transforms: Callable = None,
    ) -> None:
        assert mode in POSSIBLE_MODES

        self.mode = mode
        self.data_dir = BASE_DIR / data_dir
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.files = list(self.data_dir.glob("unprocessed/*.jsonl.zst"))

        if not (self.data_dir / "sentences.zst").exists():
            self._process_data()

        self.num_passages, self.num_sentences = self._read_info_file()

        self.current_idx = 0

        # necessry bc apparently you cant prev an iter
        if self.mode == "sentence_to_sentence":
            self.previous_sentence = None

        self.sentence_gen = self._read_jsonl(self.data_dir / "sentences.zst")

    def __len__(self: Self) -> int:
        return (
            self.num_sentences - self.num_passages
            if self.mode == "sentence_to_sentence"
            else self.num_passages
        )

    # it's gonna be very slow implementing this to go idx by idx
    # might just load the next sentence / paragraph off the dataset
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

    def _read_jsonl(self: Self, file) -> Iterator[str]:
        zstd_ctx = zstd.ZstdDecompressor()
        with open(file, "rb") as f:
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
            open(self.data_dir / "sentences.zst", "wb+") as f,
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

        with open(self.data_dir / "info.json", "w+") as f:
            json.dump(info_json, f)

    def _read_info_file(self: Self) -> tuple:
        with open(self.data_dir / "info.json", "r") as f:
            data = json.load(f)

        num_passages = data["num_passages"]
        num_sentences = data["num_sentences"]

        return num_passages, num_sentences


if __name__ == "__main__":
    c = OpenWebTextDataset()
