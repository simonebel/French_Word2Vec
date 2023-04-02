import pickle as pkl
from pathlib import Path
from typing import List

from data.tokenizer import tokenize


class Corpus:
    """
    An iterator that yields sentences, in memory friendly way
    """

    def __init__(self, data: List[str]) -> None:
        self.data = data

    def __iter__(self) -> None:
        for sent in self.data:
            yield sent


def load_data(datasets: List[str], data_dir: str) -> Corpus:
    """
    Load our dataset either from file and tokenized or from a dump.
    """

    pkl_path = Path(data_dir).joinpath("corpus.pkl")

    if not pkl_path.exists():
        data = []
        for dataset in datasets:
            dataset_dir = Path(data_dir, "processed", dataset)
            files = [file for file in dataset_dir.iterdir() if file.is_file()]
            for file in files:
                with open(file, "r") as f:
                    data.extend(f.read().splitlines())

        sentences = tokenize(data)

        with open(pkl_path, "wb") as f:
            pkl.dump(sentences, f)
    else:
        with open(pkl_path, "rb") as f:
            sentences = pkl.load(f)

    return Corpus(sentences)
