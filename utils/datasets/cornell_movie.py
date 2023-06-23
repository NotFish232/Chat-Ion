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
        processed_file_name: str = "processed.json",
        max_context_length: int = 256,
        max_sentence_length: int = 64,
        transforms: Callable = None,
        target_transforms: Callable = None,
    ) -> None:
        self.data_dir = DATA_DIR / folder_name
        self.unprocessed_file_name = unprocessed_file_name
        self.processed_file_name = processed_file_name

        self.max_context_length = max_context_length
        self.max_sentence_length = max_sentence_length

        self.transforms = transforms
        self.target_transforms = target_transforms

        self.vocab = Vocabulary()

        if not (self.data_dir / processed_file_name).exists():
            self.conversations = self._process_data()
            self._save_data()
        else:
            self.conversations = self._load_data()

    def _pad_sentence(self: Self, words: list[str], n: int) -> list[int]:
        words = words[: self.max_sentence_length]

        words.extend(self.vocab.PAD_IDX for _ in range(n - len(words)))

        return words

    def __len__(self: Self) -> int:
        return len(self.conversations)

    def __getitem__(self: Self, idx: int) -> tuple:
        conversation = next(
            self.conversations[i]
            for i in range(len(self.conversations) - 1)
            if self.conversations[i + 1]["idx"] > idx
        )
        sentences = conversation["sentences"]
        sentence_idx = idx - conversation["idx"] + 1

        question = self._pad_sentence(
            [self.vocab.CLS_IDX]
            + sum(
                (
                    self.vocab.tokenize(q) + [self.vocab.SEP_IDX]
                    for q in sentences[:sentence_idx]
                ),
                [],
            )[:-1],
            self.max_context_length,
        )
        answer = self._pad_sentence(
            [self.vocab.SOS_IDX]
            + self.vocab.tokenize(sentences[sentence_idx])
            + [self.vocab.EOS_IDX],
            self.max_sentence_length,
        )

        if self.transforms is not None:
            question = self.transforms(question)
        if self.target_transforms is not None:
            answer = self.target_transforms(answer)

        return question, answer

    def _load_data(self: Self) -> list[dict]:
        with open(self.data_dir / self.processed_file_name, "r") as f:
            data = json.load(f)
        return data

    def _process_data(self: Self) -> list[dict]:
        _conversations = {}
        with open(self.data_dir / self.unprocessed_file_name, "r") as f:
            for line in f:
                j = json.loads(line)
                conv_id = j["conversation_id"]
                if conv_id not in _conversations:
                    _conversations[conv_id] = []
                _conversations[conv_id].insert(0, j["text"])

        conversation_idxs = [0]

        for conv, _ in zip(_conversations.values(), range(len(_conversations) - 1)):
            conversation_idxs.append(conversation_idxs[-1] + len(conv) - 1)

        conversations = []

        for idx, sentences in zip(conversation_idxs, _conversations.values()):
            conversations.append({"idx": idx, "sentences": sentences})

        return conversations

    def _save_data(self: Self) -> None:
        with open(self.data_dir / self.processed_file_name, "w+") as f:
            json.dump(self.conversations, f)


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
