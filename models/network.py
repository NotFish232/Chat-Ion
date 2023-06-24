import math

import torch as T
from torch import nn
from torch.nn import functional as F
from typing_extensions import Self


class PositionalEncoding(nn.Module):
    def __init__(self: Self, max_seq_len: int, embed_dim: int, dropout: float) -> None:
        super().__init__()

        assert embed_dim % 2 == 0, "Size of embeddings must be an even number"
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        encoding = T.zeros((max_seq_len, embed_dim))
        positions = T.arange(max_seq_len).unsqueeze(1).repeat(1, embed_dim // 2)
        div_terms = (
            (10_000 ** (T.arange(0, embed_dim, 2) / embed_dim))
            .unsqueeze(0)
            .repeat_interleave(max_seq_len, 0)
        )
        encoding[:, 0::2] = T.sin(positions / div_terms)
        encoding[:, 1::2] = T.cos(positions / div_terms)

        self.register_buffer("encoding", encoding, persistent=False)

        self.dropout = nn.Dropout1d(dropout)

    def forward(self: Self, x: T.Tensor) -> T.Tensor:
        return self.dropout(x + self.encoding[: x.size(-2)].unsqueeze(0))


class Network(nn.Module):
    def __init__(
        self: Self,
        num_embed: int,
        embed_dim: int,
        max_seq_len: int,
        num_heads: int,
        num_enc_layers: int,
        num_dec_layers: int,
        feed_forward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.embed_dim_sqrt = math.sqrt(embed_dim)

        self.embedding = nn.Embedding(num_embed, embed_dim)
        self.positional_encoding = PositionalEncoding(max_seq_len, embed_dim, dropout)
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_enc_layers,
            num_decoder_layers=num_dec_layers,
            dim_feedforward=feed_forward_dim,
            dropout=dropout,
            activation=F.gelu,
            batch_first=True,
        )
        self.linear = nn.Linear(embed_dim, num_embed)

    def forward(self: Self, src: T.Tensor, tgt: T.Tensor, **kwargs) -> T.Tensor:
        src = self.embedding(src) * self.embed_dim_sqrt
        tgt = self.embedding(tgt) * self.embed_dim_sqrt

        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        x = self.transformer(src, tgt, **kwargs)

        x = self.linear(x)

        if not self.training:
            x = F.softmax(x, dim=-1)

        return x


def main() -> None:
    n = Network(10, 8)
    n.eval()
    print(f"{sum(i.numel() for i in n.parameters()):,}")
    x = T.randint(0, 5, (1, 10))
    y = T.randint(0, 5, (1, 10))
    with T.no_grad():
        print(x, y, T.argmax(n(x, y), dim=1), sep="\n")


if __name__ == "__main__":
    main()
