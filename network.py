import torch as T
from torch import nn
from typing_extensions import Self


class PositionalEncoding(nn.Module):
    def __init__(self: Self, num_embed: int, embed_dim: int) -> None:
        super().__init__()

        assert embed_dim % 2 == 0, "Size of embeddings must be an even number"
        self.num_embed = num_embed
        self.embed_dim = embed_dim

        self.encoding = T.zeros((num_embed, embed_dim))
        self.dropout = nn.Dropout2d()

   
        positions = T.arange(num_embed).unsqueeze(1).repeat(1, embed_dim // 2)
        div_terms = (
            (10_000 ** (T.arange(0, embed_dim, 2) / embed_dim))
            .unsqueeze(0)
            .repeat_interleave(num_embed, 0)
        )
        self.encoding[:, 0::2] = T.sin(positions / div_terms)
        self.encoding[:, 1::2] = T.cos(positions / div_terms)

    def forward(self: Self, x: T.Tensor) -> T.Tensor:
        return self.dropout(x + self.encoding[:x.size(0)])


class Network(nn.Module):
    def __init__(self: Self, num_embed: int, embed_dim: int) -> None:
        self.embeddings = nn.Sequential(
            nn.Embedding(num_embed, embed_dim),
        )
        self.transformer = nn.Transformer()

    def forward(self: Self, x: T.Tensor) -> T.Tensor:
        pass


def main() -> None:
    p = PositionalEncoding(100, 10_000)
    print(p.embeddings)


if __name__ == "__main__":
    main()
