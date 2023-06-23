from pathlib import Path

import torch as T
from torch import nn, optim
from torch.optim import lr_scheduler
from typing_extensions import Self

BASE_DIR = Path(__file__).parents[1]


class ModelLoader:
    def __init__(
        self: Self,
        checkpoint_dir: str = "./checkpoints",
        model_dir: str = "./trained_models",
        preserve_old: bool = True,
    ) -> None:
        self.checkpoint_dir = BASE_DIR / checkpoint_dir
        self.model_dir = BASE_DIR / model_dir
        self.preserve_old = preserve_old

    @staticmethod
    def extract_accuracy(model: Path) -> float:
        return float(model.stem.split("-")[-1])

    def checkpoint_exists(self: Self) -> bool:
        return len(list(self.checkpoint_dir.glob("*.pt"))) != 0

    def model_exists(self: Self) -> bool:
        return Path(self.model_dir / "model.pt").exists()

    def load_model(self: Self) -> tuple:
        model_path = self.model_dir / "model.pt"
        return T.load(model_path) if self.model_exists() else None

    def load_checkpoint(self: Self) -> tuple:
        if not self.checkpoint_exists():
            return None

        models = list(self.checkpoint_dir.glob("*.pt"))

        # find model with highest accuracy
        highest_acc_model = max(models, key=self.extract_accuracy)

        return T.load(self.checkpoint_dir / highest_acc_model)

    def save_checkpoint(
        self: Self,
        network: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: lr_scheduler.LRScheduler,
        epochs: int,
        accuracy: float,
    ) -> None:
        if not self.preserve_old:
            for file in self.checkpoint_dir.glob("*.pt"):
                file.unlink()

        T.save(
            {
                "network": network.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epochs": epochs,
                "accuracy": accuracy,
            },
            f"{self.checkpoint_dir}/checkpoint-{epochs}-{100 * accuracy:05.2f}.pt",
        )

    def save_model(self: Self, network: nn.Module) -> None:
        T.save(network.state_dict(), f"{self.model_dir}/model.pt")
