from pathlib import Path
from enum import Enum

BASE_DIR = Path(__file__).parents[2]
DATA_DIR = BASE_DIR / "data"


"""
modes for training
pass in sentence -> model outputs sentence
pass in sentence -> model outputs rest of passage
pass in most of passage -> model outputs last sentence
pass in half the passage -> model outputs other half of passage
masks random tokens, like in BERT
"""


class Modes(Enum):
    SentToSent = 0
    SentToPass = 1
    PassToSent = 2
    PassToPass = 3
    Masking = 4
    Conversation = 5
