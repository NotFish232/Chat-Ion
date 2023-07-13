import torch as T
from torch import nn 
from typing_extensions import Self


class Linformer(nn.Module):
    def __init__(self: Self) -> None:
        super(Linformer, self).__init__()
    
    def forward(self: Self, src: T.Tensor, tgt: T.Tensor) -> T.Tensor:
        pass