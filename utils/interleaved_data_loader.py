from typing_extensions import Self
from torch.utils.data import DataLoader
from typing import Iterator, Any
from .datasets.shared import Modes


class InterleavedDataLoader(DataLoader):
    def __init__(self: Self, *dataloaders: tuple[DataLoader]) -> None:
        self.dataloaders = dataloaders
        self.lengths = [len(d) for d in dataloaders]
        self.iterators = [iter(d) for d in dataloaders]

        min_length = min(self.lengths)
        # how many to yield from each iterator before moving to next one
        self.counts = [length // min_length for length in self.lengths]

    @property
    def current_mode(self: Self) -> Modes:
        return self.dataloaders[self.current_idx].dataset.mode

    def __len__(self: Self) -> int:
        return sum(self.lengths)

    def __iter__(self: Self) -> Iterator:
        self.current_idx = 0
        self.current_count = 0
        return self

    def __next__(self: Self) -> Any:
        if all(i is None for i in self.iterators):
            self.iterators = [iter(d) for d in self.dataloaders]
            raise StopIteration()

        while self.iterators[self.current_idx] is None:
            self.current_idx = (self.current_idx + 1) % len(self.dataloaders)
            self.current_count = 0

        try:
            ret = next(self.iterators[self.current_idx])
            self.current_count += 1
            if self.current_count == self.counts[self.current_idx]:
                self.current_idx = (self.current_idx + 1) % len(self.dataloaders)
                self.current_count = 0
            return ret

        except StopIteration:
            self.iterators[self.current_idx] = None
            return self.__next__()
