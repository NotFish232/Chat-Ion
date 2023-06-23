import json
import os
from collections import Counter
from string import punctuation
from typing import Callable, Iterator

from torch.utils.data import Dataset
from typing_extensions import Self


class CornellMovieDataset(Dataset):
    def __init__(
        self: Self,
        file_name: str = "raw.json",
        processed_file_name: str = "processed.txt",
        data_dir: str = "data/cornell",
        max_sentence_length: int = 15,
        max_word_length: int = 10,
        min_word_freq: int = 5,
        transforms: Callable = None,
        target_transforms: Callable = None,
    ) -> None:
        self.data_dir = data_dir
        self.max_sentence_length = max_sentence_length
        self.max_word_length = max_word_length
        self.min_word_freq = min_word_freq

        self.transforms = transforms
        self.target_transforms = target_transforms

        should_process_data = not os.path.isfile(data_dir + processed_file_name)

        self.conversations = (
            self._process_data(file_name)
            if should_process_data
            else self._load_data(processed_file_name)
        )

        self.vocab, self.rvocab, self.vocab_count = self._build_vocab()
        self.PAD_IDX = self.vocab["<pad>"]
        self.SOS_IDX = self.vocab["<sos>"]
        self.EOS_IDX = self.vocab["<eos>"]
        self.OUT_OF_VOCAB_IDX = self.vocab["<oov>"]

        if should_process_data:
            self._filter_conversations_by_vocab()
            self._save_data(processed_file_name)

    @property
    def num_words(self: Self):
        return len(self.vocab)

    def tokenize_sentence(
        self: Self, words: list[str], add_start_and_end: bool = False
    ) -> list[int]:
        word_idxs = [self.vocab.get(w, self.OUT_OF_VOCAB_IDX) for w in words]
        padding = self.max_sentence_length - len(word_idxs)

        if add_start_and_end:
            word_idxs.insert(0, self.SOS_IDX)
            word_idxs.append(self.EOS_IDX)

        word_idxs.extend(self.PAD_IDX for _ in range(padding))

        return word_idxs

    def __len__(self: Self) -> int:
        return len(self.conversations)

    def __getitem__(self: Self, idx: int) -> tuple:
        question, answer = self.conversations[idx]
        question_idxs = self.tokenize_sentence(question)
        answer_idxs = self.tokenize_sentence(answer, add_start_and_end=True)

        if self.transforms is not None:
            question_idxs = self.transforms(question_idxs)
        if self.target_transforms is not None:
            answer_idxs = self.target_transforms(answer_idxs)

        return question_idxs, answer_idxs

    def _vocab_generator(self: Self) -> Iterator[str]:
        for conv in self.conversations:
            question, answer = conv
            for word in question + answer:
                if len(word) <= self.max_word_length:
                    yield word
        special_tokens = ["<pad>", "<sos>", "<eos>", "<oov>"]
        for token in special_tokens:
            yield token

    def _build_vocab(self: Self) -> tuple[dict, dict]:
        counter = Counter(self._vocab_generator()).items()
        vocab, rvocab, vocab_count = {}, {}, {}
        for i, (el, count) in enumerate(counter):
            vocab[el] = i
            rvocab[i] = el
            vocab_count[el] = count
        return vocab, rvocab, vocab_count

    def _load_data(self: Self, processed_file_name: str) -> list[tuple[list[str]]]:
        conversations = []
        with open(f"{self.data_dir}/{processed_file_name}", "r") as f:
            for line in f:
                line = line.strip()
                question, answer = line.split(" ### ")
                conversations.append((question.split(" "), answer.split(" ")))
        return conversations

    def _process_data(self: Self, file_name: str) -> list[tuple[list[str]]]:
        _conversations = {}
        table = dict.fromkeys(map(ord, punctuation), " ")
        with open(self.data_dir + file_name, "r") as f:
            for line in f:
                j = json.loads(line)
                conv_id = j["conversation_id"]
                if conv_id not in _conversations:
                    _conversations[conv_id] = []
                text = j["text"].translate(table).lower()
                _conversations[conv_id].insert(0, text.split())

        conversations = []
        for conv in _conversations.values():
            for question, answer in zip(conv, conv[1:]):
                if (
                    0 < len(question) <= self.max_sentence_length
                    and 0 < len(answer) <= self.max_sentence_length
                ):
                    conversations.append((question, answer))

        return conversations

    def _save_data(self: Self, processed_file_name: str) -> None:
        with open(self.data_dir + processed_file_name, "w+") as f:
            for conv in self.conversations:
                question, answer = conv
                f.write(f"{' '.join(question)} ### {' '.join(answer)} \n")

    def _filter_conversations_by_vocab(self: Self) -> None:
        new_conversations = []
        for conv in self.conversations:
            question, answer = conv
            if all(
                w in self.vocab_count and self.vocab_count[w] > self.min_word_freq
                for w in " ".join(question + answer).split(" ")
            ):
                new_conversations.append(conv)

        self.conversations = new_conversations

        # in order to not mess up idxs in training (i.e. different idxs when loading 1st / 2nd time)
        self.vocab, self.rvocab, self.vocab_count = self._build_vocab()


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
