import math
import timeit

import torch as T
from torch import nn
from typing_extensions import Self


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(
        self: Self, max_seq_len: int, embed_dim: int, dropout_p: float
    ) -> None:
        super(SinusoidalPositionalEncoding, self).__init__()

        assert embed_dim % 2 == 0, "Embeddings dimension must be even"
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        encodings = T.zeros((max_seq_len, embed_dim))
        positions = T.arange(max_seq_len).unsqueeze(1).repeat(1, embed_dim // 2)
        div_terms = (
            (10_000 ** (T.arange(0, embed_dim, 2) / embed_dim))
            .unsqueeze(0)
            .repeat_interleave(max_seq_len, 0)
        )
        encodings[:, 0::2] = T.sin(positions / div_terms)
        encodings[:, 1::2] = T.cos(positions / div_terms)

        self.register_buffer("encodings", encodings, persistent=False)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self: Self, x: T.Tensor) -> T.Tensor:
        seq_len = x.size(1)
        encodings = self.encodings[:seq_len].unsqueeze(0)
        return self.dropout(encodings)


class LearnedPositionalEncoding(nn.Module):
    def __init__(
        self: Self, max_seq_len: int, embed_dim: int, dropout_p: float
    ) -> None:
        super(LearnedPositionalEncoding, self).__init__()

        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        self.encodings = nn.Embedding(max_seq_len, embed_dim)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self: Self, x: T.Tensor) -> T.Tensor:
        seq_len = x.size(1)
        positions = T.arange(seq_len, device=x.device)
        encodings = self.encodings(positions).unsqueeze(0)
        return self.dropout(encodings)


class AxialPositionalEncoding(nn.Module):
    def __init__(
        self: Self, max_seq_len: int, embed_dim: int, dropout_p: float
    ) -> None:
        super(AxialPositionalEncoding, self).__init__()

        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        self.d = (embed_dim // 2, (embed_dim + 1) // 2)
        self.n = self._get_largest_factor_pair(max_seq_len)

        self.x1 = nn.Parameter(T.randn(self.n[0], self.d[0]))
        self.x2 = nn.Parameter(T.randn(self.n[1], self.d[1]))

        self.dropout = nn.Dropout(dropout_p)

    def _get_largest_factor_pair(self: Self, n: int) -> tuple[int, int]:
        for i in reversed(range(int(math.sqrt(n) + 1))):
            if n % i == 0:
                return i, n // i
        return -1, -1

    def forward(self: Self, x: T.Tensor) -> T.Tensor:
        seq_len = x.size(1)
        encodings = (
            T.arange(self.embed_dim, device=x.device, dtype=x.dtype)
            .view(1, 1, -1)
            .repeat_interleave(seq_len, 1)
        )

        seq_positions = T.arange(seq_len, device=x.device)
        x1_part = self.x1[seq_positions % self.n[0]].flatten()
        x2_part = self.x2[seq_positions // self.n[0]].flatten()

        encodings_bool = encodings < self.d[0]
        encodings[encodings_bool] = x1_part
        encodings[~encodings_bool] = x2_part

        return self.dropout(encodings)


if __name__ == "__main__":
    seq_len = 1028
    embed_dim = 768
    x = T.randn(128, seq_len, embed_dim)
    pe1 = SinusoidalPositionalEncoding(seq_len, embed_dim, 0)
    pe2 = LearnedPositionalEncoding(seq_len, embed_dim, 0)
    pe3 = AxialPositionalEncoding(seq_len, embed_dim, 0)
    print(f"pe1: {sum(i.numel() for i in pe1.parameters()):,} params")
    print(f"pe2: {sum(i.numel() for i in pe2.parameters()):,} params")
    print(f"pe3: {sum(i.numel() for i in pe3.parameters()):,} params")

    with T.no_grad():
        t1 = timeit.timeit(lambda: pe1(x), number=1_000)
        t2 = timeit.timeit(lambda: pe2(x), number=1_000)
        t3 = timeit.timeit(lambda: pe3(x), number=1_000)

    print(f"pe1: {t1:.2f} sec")
    print(f"pe2: {t2:.2f} sec")
    print(f"pe3: {t3:.3f} sec")
