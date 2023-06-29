import torch as T
from torch import nn
from typing_extensions import Self


class MultiHeadAttention(nn.Module):
    def __init__(self: Self, embed_dim: int, num_heads: int, dropout_p: float) -> None:
        super(MultiHeadAttention, self).__init__()

        assert (
            embed_dim % num_heads == 0
        ), "Embeddings dimension must be divisible by num heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.embed_dim_per_head = embed_dim // num_heads

        self.key_linear = nn.Linear(embed_dim, embed_dim)
        self.query_linear = nn.Linear(embed_dim, embed_dim)
        self.value_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(
        self: Self,
        Q: T.Tensor,
        K: T.Tensor,
        V: T.Tensor,
        key_padding_mask: T.Tensor,
        attn_mask: T.Tensor = None,
    ) -> T.Tensor:
        ...


class Reformer(nn.Module):
    def __init__(self: Self, embed_dim: int, num_heads: int, dropout_p: float) -> None:
        super(Reformer, Self).__init__()

    def forward(self: Self, x: T.Tensor) -> T.Tensor:
        ...


