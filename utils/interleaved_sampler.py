from typing import Iterator

from torch.utils.data.distributed import Sampler
from typing_extensions import Self


class InterleavedSampler(Sampler):
    def __init__(self: Self, dataset_len: int, rank: int, world_size: int) -> None:
        self.indices = range(rank, dataset_len, world_size)
        self.length = dataset_len // world_size

    def __iter__(self: Self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self: Self) -> int:
        return self.length
