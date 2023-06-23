from pathlib import Path

import torch as T
from torch import nn, optim
from torch.cuda import amp
from torch.optim import lr_scheduler
from typing_extensions import Self
import json

BASE_DIR = Path(__file__).parents[1]


DEFAULT_SAVE_KEYS = [
    "network",
    "optimizer",
    "scheduler",
    "scaler",
    "epochs",
    "accuracy",
]


class ModelManager:
    def __init__(
        self: Self,
        model_name: str,
        save_keys: list[str] = DEFAULT_SAVE_KEYS,
        checkpoint_dir: str = "checkpoints",
        model_dir: str = "trained_models",
    ) -> None:
        self.model_name = model_name
        self.save_keys = save_keys
        self.checkpoint_dir = BASE_DIR / checkpoint_dir / model_name
        self.model_dir = BASE_DIR / model_dir / model_name

        self.checkpoint_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exists_ok=True)

    def checkpoint_exists(self: Self) -> bool:
        return len(list(self.checkpoint_dir.glob("*.pt"))) != 0

    def model_exists(self: Self) -> bool:
        return (
            Path(self.model_dir / "model.pt").exists()
            and Path(self.model_dir / "info.json").exists()
        )

    def load_model(self: Self, model: nn.Module) -> dict | None:
        if not self.model_exists():
            return None

        state_dict = T.load(self.model_dir / "model.pt")
        model.load_state_dict(state_dict)

        with open(self.model_dir / "info.json", "rt") as f:
            info = json.load(f)

        return info

    def load_checkpoint(self: Self, *kwargs: tuple) -> tuple:
        if not self.checkpoint_exists():
            return None

        checkpoints = list(self.checkpoint_dir.glob("*.pt"))

        most_recent_checkpoint = max(checkpoints, key=lambda x: x.stem.split("-")[-1])
        most_recent_checkpoint = T.load(most_recent_checkpoint)

        return_values = []
        for key, value in most_recent_checkpoint.items():
            if key in kwargs.keys():
                kwargs[key].load_state_dict(value)
            else:
                return_values.append(value)

        return tuple(return_values)

    def save_checkpoint(self: Self, *args: tuple) -> None:
        checkpoint = {}
        for key, value in zip(self.save_keys, args):
            if hasattr(value, "state_dict"):
                checkpoint[key] = value.state_dict()
            else:
                checkpoint[key] = value

        checkpoints = list(self.checkpoint_dir.glob("*.pt"))
        checkpoint_num = len(checkpoints) + 1

        T.save(
            checkpoint,
            self.checkpoint_dir / f"checkpoint-{checkpoint_num}.pt",
        )

    def save_model(self: Self, network: nn.Module, model_kwargs: dict) -> None:
        T.save(network.state_dict(), f"{self.model_dir}/model.pt")
        with open(self.model_dir / "info.json", "wt+") as f:
            json.dump(model_kwargs, f)
