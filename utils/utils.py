import string
from .datasets.vocabulary import Vocabulary, SPECIAL_TOKENS

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


# should not insert space on left or right
NO_SPACE_PUNCTUATION = ["'", '"', "`", "-", "_", "/", "\\", "|"]
# should insert one space on the left
LEFT_SPACE_PUNCTUATION = ["(", "{", "[", "$"]
# should insert one space on the right
RIGHT_SPACE_PUNCTUATION = [".", ",", "?", "!", ";", ":", ")", "}", "]", "%"]
# should insert space on left and right
TWO_SPACE_PUNCTUATION = ["@", "#", "^", "&", "*", "~", "+", "=", "<", ">"]


def join_tokens(tokens: list[int | str], subword_start: str = "##") -> str:
    if isinstance(tokens[0], int):
        v = Vocabulary()
        tokens = [v.idx_to_token.get(i, v.OOV_IDX) for i in tokens]

    s = " ".join(token for token in tokens if token not in SPECIAL_TOKENS) + " "
    s = s.replace(" " + subword_start, "")

    for punct in string.punctuation:
        if punct in TWO_SPACE_PUNCTUATION:
            continue

        punct_replaced = punct

        if punct in LEFT_SPACE_PUNCTUATION:
            punct_replaced = " " + punct_replaced
        if punct in RIGHT_SPACE_PUNCTUATION:
            punct_replaced = punct_replaced + " "

        s = s.replace(" " + punct + " ", punct_replaced)

    return s.strip()
