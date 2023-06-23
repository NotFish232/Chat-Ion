import torch as T


def make_look_ahead_mask(n: int, device: T.device) -> T.Tensor:
    return T.triu(
        T.ones(
            n,
            n,
            dtype=T.bool,
            device=device,
        ),
        diagonal=1,
    )
