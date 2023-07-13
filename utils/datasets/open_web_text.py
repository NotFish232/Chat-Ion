import io
import json
import multiprocessing as mp
import random
from typing import Callable, Iterator
import numpy as np
import zstandard as zstd
from torch.utils.data import Dataset
from tqdm import tqdm
from typing_extensions import Self
from typing import Any

from .shared import DATA_DIR, Modes
from .vocabulary import Vocabulary

MASKING_PROB = 0.15
RANDOM_WORD_PROB = 0.1
ACTUAL_WORD_PROB = 0.1


class OpenWebTextDataset(Dataset):
    def __init__(
        self: Self,
        mode: Modes | str,
        folder_name: str = "openwebtext2",
        unprocessed_folder_name: str = "unprocessed",
        processed_file_name: str = "processed.bin",
        info_file_name: str = "info.json",
        max_sentence_length: int = 256,
        max_passage_length: int = 1024,
        max_processed_length: int = 2048,
        transforms: Callable = None,
        target_transforms: Callable = None,
    ) -> None:
        if isinstance(mode, str):
            mode = Modes.__members__.get(mode)
        self.mode = mode

        self.unprocessed_folder = DATA_DIR / folder_name / unprocessed_folder_name
        self.processed_file = DATA_DIR / folder_name / processed_file_name
        self.info_file = DATA_DIR / folder_name / info_file_name

        self.max_sentence_length = max_sentence_length
        self.max_passage_length = max_passage_length
        self.max_processed_length = max_processed_length

        self.transforms = transforms
        self.target_transforms = target_transforms

        self.vocab = Vocabulary()

        self.unprocessed_files = list(self.unprocessed_folder.glob("*.jsonl.zst"))

        if not self.processed_file.exists():
            self._process_data()

        self.num_passages = self._read_info_file()

        self.data = np.memmap(
            self.processed_file,
            dtype=np.uint16,
            mode="r",
            shape=(self.num_passages, self.max_processed_length),
        )

    def __len__(self: Self) -> int:
        return self.num_passages

    def _make_sent_to_sent_task(
        self: Self, passage: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        middle = len(passage) // 2
        source = passage[:middle]
        target = passage[middle:]

        source = self.vocab.fix_length(
            source,
            self.max_sentence_length,
            add_cls_and_sep=True,
            truncate_from_left=True,
        )
        target = self.vocab.fix_length(
            target, self.max_sentence_length + 1, add_sos_and_eos=True
        )

        return source, target

    def _make_sent_to_pass_task(
        self: Self, passage: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        middle = len(passage) // 2
        source = passage[:middle]
        target = passage[middle:]

        source = self.vocab.fix_length(
            source,
            self.max_sentence_length,
            add_cls_and_sep=True,
            truncate_from_left=True,
        )
        target = self.vocab.fix_length(
            target, self.max_passage_length + 1, add_sos_and_eos=True
        )

        return source, target

    def _make_pass_to_sent_task(
        self: Self, passage: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        middle = len(passage) // 2
        source = passage[:middle]
        target = passage[middle:]

        source = self.vocab.fix_length(
            source,
            self.max_passage_length,
            add_cls_and_sep=True,
            truncate_from_left=True,
        )
        target = self.vocab.fix_length(
            target, self.max_sentence_length + 1, add_sos_and_eos=True
        )

        return source, target

    def _make_pass_to_pass_task(
        self: Self, passage: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        middle = len(passage) // 2
        source = passage[:middle]
        target = passage[middle:]

        source = self.vocab.fix_length(
            source,
            self.max_passage_length,
            add_cls_and_sep=True,
            truncate_from_left=True,
        )
        target = self.vocab.fix_length(
            target, self.max_passage_length + 1, add_sos_and_eos=True
        )

        return source, target

    def _make_masking_task(self: Self, passage: list[int]) -> tuple[list[int]]:
        source = self.vocab.fix_length(
            passage, self.max_passage_length, add_cls_and_sep=True
        )
        target = [self.vocab.PAD_IDX for _ in range(self.max_passage_length)]
        for idx in range(1, len(source)):
            if source[idx] == self.vocab.SEP_IDX:
                break
            
            prob = random.random()
            if prob <= MASKING_PROB:
                prob /= MASKING_PROB
                target[idx] = source[idx]
                if prob < RANDOM_WORD_PROB:
                    source[idx] = random.randint(0, self.vocab.num_reg_tokens - 1)
                elif prob < 1 - ACTUAL_WORD_PROB:
                    source[idx] = self.vocab.MASK_IDX

        return source, target

    def __getitem__(self: Self, idx: int) -> tuple[Any, Any]:
        passage = self.data[idx].tolist()

        make_task_func = (
            self._make_sent_to_sent_task
            if self.mode == Modes.SentToSent
            else self._make_sent_to_pass_task
            if self.mode == Modes.SentToPass
            else self._make_pass_to_sent_task
            if self.mode == Modes.PassToSent
            else self._make_pass_to_pass_task
            if self.mode == Modes.PassToPass
            else self._make_masking_task
            if Modes.Masking
            else None
        )
        source, target = make_task_func(passage)

        if self.transforms is not None:
            source = self.transforms(source)
        if self.target_transforms is not None:
            target = self.target_transforms(target)

        return {"src": source, "tgt": target, "mode": self.mode}

    def _mask(
        self: Self, input: list[str], target: list[str]
    ) -> tuple[list[int], list[int]]:
        # don't mask cls or sep tokens
        for idx in range(1, len(input) - 1):
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

    def _read_jsonl(self: Self, file_path: str) -> Iterator[str]:
        zstd_ctx = zstd.ZstdDecompressor()
        with open(file_path, "rb") as f:
            reader = io.BufferedReader(zstd_ctx.stream_reader(f))
            for line in reader:
                yield json.loads(line)["text"]

    def _process_file(self: Self, file: str) -> list[int]:
        passages = []
        for passage in self._read_jsonl(file):
            passage_tokens = self.vocab.tokenize(passage, self.max_processed_length)
            passages.append(passage_tokens)
        passages = np.array(passages, dtype=np.uint16)
        return passages

    def _process_data(self: Self) -> dict:
        num_passages = 0
        for file in tqdm(self.unprocessed_files, desc=f"Finding length"):
            for _ in self._read_jsonl(file):
                num_passages += 1

        array = np.memmap(
            filename=self.processed_file,
            mode="w+",
            dtype=np.uint16,
            shape=(num_passages, self.max_processed_length),
        )
        """
        num_processes = 8
        with mp.Pool(num_processes) as p:
            m = p.imap(self._process_file, self.unprocessed_files)
            progress_bar = tqdm(
                m, total=len(self.unprocessed_files), desc="Processing data.."
            )
            idx = 0
            for passages in progress_bar:
                array[idx : idx + len(passages)] = passages
                idx += len(passages)
        """
        idx = 0
        for file in tqdm(self.unprocessed_files, desc=f"Writing passages"):
            new_passages = self._process_file(file)
            array[idx : idx + len(new_passages)] = new_passages
            idx += len(new_passages)
        array.flush()

        with open(self.info_file, "w+") as f:
            f.write(json.dumps({"num_passages": len(array)}))

    def _read_info_file(self: Self) -> tuple:
        with open(self.info_file, "r") as f:
            data = json.load(f)

        return data["num_passages"]


if __name__ == "__main__":
    c = OpenWebTextDataset()
