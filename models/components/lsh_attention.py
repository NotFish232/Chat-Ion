from torch import nn
from typing_extensions import Self
import torch as T
import random


class LSHAttention(nn.Module):
    def __init__(self: Self) -> None:
        pass


def shingle(l: str, k: int) -> list[str]:
    s = set()
    for i in range(len(l) - k + 1):
        s.add(l[i : i + k])
    return s


def one_hot(l: list[str], v: list[str]) -> list[int]:
    lst = []
    for i in v:
        lst.append(1 if i in l else 0)
    return lst


def hash(l: list[int], n: int) -> list[int]:
    out = []
    idxs = list(range(len(l)))
    for _ in range(n):
        random.shuffle(idxs)
        idx = next(i for i in idxs if l[i] == 1)
        out.append(idx)

    return out


def jaccard_sim(a: list, b: list):
    a = set(a)
    b = set(b)
    return len(a & b) / len(a | b)
    a.inter

_a = "flying fish flew by the space station"
_b = "we will not allow you to bring your pet armadillo along"
_c = "he figured a few sticks of dynamite were easier than a fishing pole to catch fish"

k = 2
n = 20

a = shingle(_a, k)
b = shingle(_b, k)
c = shingle(_c, k)

vocab = a | b | c

a_hot = one_hot(a, vocab)
b_hot = one_hot(b, vocab)
c_hot = one_hot(c, vocab)

a_hash = hash(a_hot, n)
b_hash = hash(b_hot, n)
c_hash = hash(c_hot, n)


ab_sim = jaccard_sim(a_hash, b_hash)
ac_sim = jaccard_sim(a_hash, c_hash)
bc_sim = jaccard_sim(b_hash, c_hash)

print(f"{ab_sim=}", f"{ac_sim=}", f"{bc_sim=}", sep="\n")
