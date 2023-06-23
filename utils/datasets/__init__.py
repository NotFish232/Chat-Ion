from pathlib import Path

from .cornell_movie import CornellMovieDataset
from .open_web_text import OpenWebTextDataset

BASE_DIR = Path(__file__).parents[2]
DATA_DIR = BASE_DIR / "data"
