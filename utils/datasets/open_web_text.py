import io
import json
from pathlib import Path
from typing import Callable, Iterator

import zstandard as zstd
from torch.utils.data import Dataset
from typing_extensions import Self

base_dir = Path(__file__).parents[2]


class OpenWebTextDataset(Dataset):
    def __init__(
        self: Self,
        data_dir: str = "data/openwebtext2",
        transforms: Callable = None,
        target_transforms: Callable = None,
    ) -> None:
        self.data_dir = base_dir / data_dir
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.files = list(self.data_dir.glob("*.jsonl.zst"))
        self.zstd_ctx = zstd.ZstdDeCompressor()

        if not (self.data_dir / "info.json").exists():
            self._make_info_file()

        self.info = self._read_info_file()

    def read_jsonl(self: Self, file) -> Iterator[str]:
        with open(file, "rb") as f:
            reader = io.BufferedReader(self.zstd_ctx.stream_reader(f))
            for line in reader:
                json_line = json.loads(line)
                yield json_line["text"]

    def _make_info_file(self: Self) -> None:
        pass

    def _read_info_file(self: Self) -> dict:
        pass


c = OpenWebTextDataset()
