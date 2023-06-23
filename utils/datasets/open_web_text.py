from torch.utils.data import Dataset
from typing_extensions import Self 
from typing import Callable
from pathlib import Path

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

      self.files = self.data_dir.glob("*.jsonl.zst")
      print(len([i for i in self.files]))

c = OpenWebTextDataset()