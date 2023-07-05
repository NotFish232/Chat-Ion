from torch import nn
from torch.nn import functional as F
from typing_extensions import Self
import torch as T
import random
import math


class MultiheadLSHAttention(nn.Module):
    def __init__(
        self: Self, embed_dim: int, num_heads: int = 8, num_hashes: int = 8
    ) -> None:
        super(MultiheadLSHAttention, self).__init__()

        assert num_hashes % 2 == 0, "Number of hashes must be even"
        assert (
            embed_dim % num_heads == 0
        ), "Number of heads must divide evenly into embed dim"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_embed_dim = embed_dim // num_heads
        self.num_hashes = num_hashes

        self.qk_projection = nn.Linear(embed_dim, embed_dim)
        self.v_projection = nn.Linear(embed_dim, embed_dim)
        self.out_projection = nn.Linear(embed_dim, embed_dim)

        hash_matrix = T.randn((self.head_embed_dim, num_hashes // 2))
        self.register_buffer("hash_matrix", hash_matrix, persistent=False)

    def _split_heads(self: Self, x: T.Tensor) -> T.Tensor:
        # (batch_size, seq_len, embed_dim) -> (batch_size, num_heads, seq_len, embed_dim_per_head)
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, self.num_heads, seq_len, self.head_embed_dim)
        return x

    def _merge_heads(self: Self, x: T.Tensor) -> T.Tensor:
        # (batch_size, num_heads, seq_len, embed_dim_per_head) -> (batch_size, seq_len, embed_dim)
        batch_size, _, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.embed_dim)
        return x

    def _attention(self: Self, Q: T.Tensor, K: T.Tensor, V: T.Tensor) -> T.Tensor:
        # Q: (batch_size, num_heads, q_seq_len, embed_dim_per_head)
        # K: (batch_size, num_heads, embed_dim_per_head, k_seq_len)
        K = K.transpose(-2, -1)

        # attention_matrix: (batch_size, num_heads, q_seq_len, k_seq_len)
        print(self.hash_matrix.shape, Q.shape)
        print(self.hash_matrix * Q)
        attention_matrix = (Q @ K) / math.sqrt(self.head_embed_dim)
        attention_matrix = F.softmax(attention_matrix, dim=-1)

        # output: (batch_size, num_heads, v_seq_len)
        output = attention_matrix @ V

        return output

    def forward(self: Self, Q: T.Tensor, K: T.Tensor, V: T.Tensor) -> T.Tensor:
        Q = self.qk_projection(Q)
        K = self.qk_projection(K)
        V = self.v_projection(V)

        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        output = self._attention(Q, K, V)
        output = self._merge_heads(output)
        output = self.out_projection(output)

        return output


class LSHAttention(nn.Module):
    def __init__(self: Self, embed_dim: int, num_hashes: int = 8) -> None:
        super(LSHAttention, self).__init__()

        assert num_hashes % 2 == 0, "Number of hashes must be even"
        self.embed_dim = embed_dim
        self.num_hashes = num_hashes

        self.qk_projection = nn.Linear(embed_dim, embed_dim)
        self.v_projection = nn.Linear(embed_dim, embed_dim)
        self.out_projection = nn.Linear(embed_dim, embed_dim)


    def _attention(self: Self, Q: T.Tensor, K: T.Tensor, V: T.Tensor) -> T.Tensor:
        # Q: (batch_size, seq_len, embed_dim)
        # K: (batch_size, embed_dim, seq_len)
        self._hash_vector(Q)


        # attention_matrix: (batch_size, num_heads, q_seq_len, k_seq_len)
        attention_matrix = (Q @ K) / math.sqrt(self.embed_dim)
        attention_matrix = F.softmax(attention_matrix, dim=-1)

        # output: (batch_size, num_heads, v_seq_len)
        output = attention_matrix @ V

        return output
    
    def _hash_vector(self: Self, x: T.Tensor) -> T.Tensor:
        batch_size, seq_len, _ = x.shape
        R = T.randn((self.embed_dim, self.num_hashes // 2), device=x.device) 
        R = R.view(1, 1, self.embed_dim, self.num_hashes // 2)
        R = R.expand(batch_size, seq_len, -1, -1)
        print(x.shape, R.shape)
        hashes = x * R

        return hashes >= 0


    def forward(self: Self, Q: T.Tensor, K: T.Tensor, V: T.Tensor) -> T.Tensor:
        Q = self.qk_projection(Q)
        K = self.qk_projection(K)
        V = self.v_projection(V)

        output = self._attention(Q, K, V)
        output = self.out_projection(output)

        return output


if __name__ == "__main__":
    seq_len = 8
    embed_dim = 12
    batch_size = 2
    q = T.randint(0, 10, (batch_size, seq_len, embed_dim), dtype=T.float32)
    k = T.randint(0, 10, (batch_size, seq_len, embed_dim), dtype=T.float32)
    v = T.randint(0, 10, (batch_size, seq_len, embed_dim), dtype=T.float32)
    n = LSHAttention(embed_dim)
    with T.no_grad():
        x = n(q, k, v)
        print(x.shape)
