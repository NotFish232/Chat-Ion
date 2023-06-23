import json
from typing import Callable

from torch.utils.data import Dataset
from typing_extensions import Self

from .shared import DATA_DIR
from .vocabulary import Vocabulary


class CornellMovieDataset(Dataset):
    def __init__(
        self: Self,
        folder_name: str = "cornell",
        unprocessed_file_name: str = "raw.json",
        processed_file_name: str = "processed.txt",
        max_sentence_length: int = 15,
        transforms: Callable = None,
        target_transforms: Callable = None,
    ) -> None:
        self.data_dir = DATA_DIR / folder_name
        self.unprocessed_file_name = unprocessed_file_name
        self.processed_file_name = processed_file_name

        self.max_sentence_length = max_sentence_length

        self.transforms = transforms
        self.target_transforms = target_transforms

        self.vocab = Vocabulary()

        if not (self.data_dir / processed_file_name).exists():
            self.conversations = self._process_data()
            self._save_data()
        else:
            self.conversations = self._load_data()

    def prepare_sentence(
        self: Self, words: list[str], add_start_and_end: bool = False
    ) -> list[int]:
        word_idxs = self.vocab.tokenize(words)
        padding = self.max_sentence_length - len(word_idxs)

        if add_start_and_end:
            word_idxs.insert(0, self.vocab.SOS_IDX)
            word_idxs.append(self.vocab.EOS_IDX)

        word_idxs.extend(self.vocab.PAD_IDX for _ in range(padding))

        return word_idxs

    def __len__(self: Self) -> int:
        return len(self.conversations)

    def __getitem__(self: Self, idx: int) -> tuple:
        question, answer = self.conversations[idx]
        question_idxs = self.prepare_sentence(question)
        answer_idxs = self.prepare_sentence(answer, add_start_and_end=True)

        if self.transforms is not None:
            question_idxs = self.transforms(question_idxs)
        if self.target_transforms is not None:
            answer_idxs = self.target_transforms(answer_idxs)

        return question_idxs, answer_idxs

    def _load_data(self: Self) -> list[tuple[str]]:
        conversations = []
        with open(self.data_dir / self.processed_file_name, "r") as f:
            for line in f:
                line = line.strip()
                question, answer = line.split(" ### ")
                conversations.append((question, answer))
        return conversations

    def _process_data(self: Self) -> list[tuple[str]]:
        _conversations = {}
        with open(self.data_dir / self.unprocessed_file_name, "r") as f:
            for line in f:
                j = json.loads(line)
                conv_id = j["conversation_id"]
                if conv_id not in _conversations:
                    _conversations[conv_id] = []
                _conversations[conv_id].insert(0, j["text"])

        conversations = []
        for conv in _conversations.values():
            for question, answer in zip(conv, conv[1:]):
                if (
                    0 < len(self.vocab.tokenize(question)) <= self.max_sentence_length
                    and 0 < len(self.vocab.tokenize(answer)) <= self.max_sentence_length
                ):
                    conversations.append((question, answer))

        return conversations

    def _save_data(self: Self) -> None:
        with open(self.data_dir / self.processed_file_name, "w+") as f:
            for conv in self.conversations:
                question, answer = conv
                f.write(f"{question} ### {answer} \n")


def main() -> None:
    x = CornellMovieDataset()
    print(len(x.conversations))
    question, answer = x[2]
    print(
        question,
        answer,
        [x.rvocab[i] for i in question],
        [x.rvocab[i] for i in answer],
        sep="\n",
    )


if __name__ == "__main__":
    main()
