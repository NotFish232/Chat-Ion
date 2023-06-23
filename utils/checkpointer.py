import torch as T
from torch import nn, optim
from typing_extensions import Self
from pathlib import Path

BASE_DIR = Path(__file__).parents[1]


class CheckPointer:
    def __init__(
        self: Self, model_dir: str = "./trained_models", preserve_old: bool = False
    ) -> None:
        self.model_dir = BASE_DIR / model_dir
        self.preserve_old = preserve_old

    @staticmethod
    def extract_accuracy(model_name: str) -> float:
        return float(model_name[-8:-3])

    def exists(self: Self) -> bool:
        return len(list(self.model_dir.glob("*.pt"))) != 0

    def load(self: Self) -> tuple:
        if not self.exists():
            return None

        models = list(self.model_dir.glob("*.pt"))

        # find model with highest accuracy
        highest_acc_model = max(models, key=lambda x: self.extract_accuracy(str(x)))

        return T.load(self.model_dir / highest_acc_model)

    def save(
        self: Self,
        network: nn.Module,
        optimizer: optim.Optimizer,
        epochs: int,
        accuracy: float,
    ) -> None:
        if not self.preserve_old:
            for file in self.model_dir.glob("*.pt"):
                file.unlink()

        T.save(
            {
                "network": network.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epochs": epochs,
                "accuracy": accuracy,
            },
            f"{self.model_dir}/model-{epochs}-{100 * accuracy:05.2f}.pt",
        )
