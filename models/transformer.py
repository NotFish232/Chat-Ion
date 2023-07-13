import math

import torch as T
from torch import nn
from torch.nn import functional as F
from typing_extensions import Self
from .components.positional_encoding import SinusoidalPositionalEncoding


class Transformer(nn.Module):
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
        super(Transformer, self).__init__()

        self.embed_dim_sqrt = math.sqrt(embed_dim)

        self.embedding = nn.Embedding(num_embed, embed_dim)
        self.positional_encoding = SinusoidalPositionalEncoding(
            max_seq_len, embed_dim, dropout
        )
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

        src += self.positional_encoding(src)
        tgt += self.positional_encoding(tgt)

        # TODO: FIXME
        del kwargs["src_key_padding_mask"] 
        del kwargs["tgt_key_padding_mask"]

        x = self.transformer(src, tgt, **kwargs)

        x = self.linear(x)

        if not self.training:
            x = F.softmax(x, dim=-1)

        return x


def main() -> None:
    n = Transformer(10, 8)
    n.eval()
    print(f"{sum(i.numel() for i in n.parameters()):,}")
    x = T.randint(0, 5, (1, 10))
    y = T.randint(0, 5, (1, 10))
    with T.no_grad():
        print(x, y, T.argmax(n(x, y), dim=1), sep="\n")


if __name__ == "__main__":
    main()
