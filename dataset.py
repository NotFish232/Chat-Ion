from torch.utils.data import Dataset
from typing_extensions import Self
from typing import Iterator, Callable, Iterable
import json
import os
from string import punctuation
from collections import Counter


class ConversationDataset(Dataset):
    def __init__(
        self: Self,
        file_name: str = "raw.json",
        processed_file_name: str = "processed.txt",
        data_dir: str = "data/",
        max_sentence_length: int = 15,
        max_word_length: int = 10,
        transforms: Callable = None,
        target_transforms: Callable = None,
    ) -> None:
        self.data_dir = data_dir
        self.max_sentence_length = max_sentence_length
        self.max_word_length = max_word_length
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.conversations = (
            self._process_data(file_name, processed_file_name)
            if not os.path.isfile(data_dir + processed_file_name)
            else self._load_data(processed_file_name)
        )
        self.vocab, self.rvocab = self._build_vocab()
        self.OUT_OF_VOCAB_IDX = self.vocab["<oov>"]
        self.PAD_IDX = self.vocab["<pad>"]
    
    @property
    def num_words(self: Self) -> int:
        return len(self.vocab)

    def __len__(self: Self) -> int:
        return len(self.conversations)

    def __getitem__(self: Self, idx: int) -> tuple:
        question, answer = self.conversations[idx]
        question_idxs = [self.vocab.get(w, self.OUT_OF_VOCAB_IDX) for w in question]
        answer_idxs = [self.vocab.get(w, self.OUT_OF_VOCAB_IDX) for w in answer]

        question_padding = self.max_sentence_length - len(question_idxs)
        answer_padding = self.max_sentence_length - len(answer_idxs)
        question_idxs.extend(self.PAD_IDX for _ in range(question_padding))
        answer_idxs.extend(self.PAD_IDX for _ in range(answer_padding))

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
        special_tokens = ["<oov>", "<pad>", "<mask>"]
        for token in special_tokens:
            yield token

    def _build_vocab(self: Self) -> tuple[dict, dict]:
        counter = Counter(self._vocab_generator()).most_common()
        vocab, rvocab = {}, {}
        for i, (el, _) in enumerate(counter):
            vocab[el] = i
            rvocab[i] = el
        return vocab, rvocab

    def _load_data(self: Self, processed_file_name: str) -> list[tuple[list[str]]]:
        conversations = []
        with open(f"{self.data_dir}/{processed_file_name}", "r") as f:
            for line in f:
                line = line.strip()
                question, answer = line.split(" ### ")
                conversations.append((question.split(" "), answer.split(" ")))
        return conversations

    def _process_data(
        self: Self, file_name: str, processed_file_name: str
    ) -> list[tuple[list[str]]]:
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

        with open(self.data_dir + processed_file_name, "w+") as f:
            for conv in conversations:
                question, answer = conv
                f.write(f"{' '.join(question)} ### {' '.join(answer)} \n")

        return conversations


def main() -> None:
    x = ConversationDataset()
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
