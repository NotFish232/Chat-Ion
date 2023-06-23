import torch as T
from torch import nn
from torch.nn import functional as F
from typing_extensions import Self
import math


class PositionalEncoding(nn.Module):
    def __init__(self: Self, num_embed: int, embed_dim: int) -> None:
        super().__init__()

        assert embed_dim % 2 == 0, "Size of embeddings must be an even number"
        self.num_embed = num_embed
        self.embed_dim = embed_dim

        self.encoding = T.zeros((num_embed, embed_dim))
        self.dropout = nn.Dropout1d()

        positions = T.arange(num_embed).unsqueeze(1).repeat(1, embed_dim // 2)
        div_terms = (
            (10_000 ** (T.arange(0, embed_dim, 2) / embed_dim))
            .unsqueeze(0)
            .repeat_interleave(num_embed, 0)
        )
        self.encoding[:, 0::2] = T.sin(positions / div_terms)
        self.encoding[:, 1::2] = T.cos(positions / div_terms)

    def forward(self: Self, x: T.Tensor) -> T.Tensor:
        return self.dropout(x + self.encoding[: x.size(0)])


class Network(nn.Module):
    def __init__(self: Self, num_embed: int, embed_dim: int) -> None:
        super().__init__()

        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(num_embed, embed_dim)
        self.positional_encoding = PositionalEncoding(num_embed, embed_dim)
        self.transformer = nn.Transformer(embed_dim, batch_first=True)
        self.linear = nn.Linear(embed_dim, num_embed)

    def forward(self: Self, src: T.Tensor, tgt: T.Tensor) -> T.Tensor:
        src = self.embedding(src) * math.sqrt(self.embed_dim)
        tgt = self.embedding(tgt) * math.sqrt(self.embed_dim)

        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        x = self.transformer(src, tgt)

        x = self.linear(x)

        if not self.training:
            x = F.softmax(x, dim=1)

        return x


def main() -> None:
    n = Network(5, 16)
    n.eval()
    print(sum(i.numel() for i in n.parameters()))
    x = T.randint(0, 5, (1, 5))
    y = T.randint(0, 5, (1, 5))
    with T.no_grad():
        print(x, y, T.argmax(n(x, y), dim=1), sep = '\n')


if __name__ == "__main__":
    main()
