from os.path import join
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from src.main.data import get_random_transposition, pad, preprocess_midi
from src.main.util import root_dir


def _augment_dataset(midi_sequences: List[np.ndarray]) -> List[np.ndarray]:
    """

    :param midi_sequences:
    :return:
    """
    augmented_sequences = []
    for sequence in tqdm(midi_sequences):
        augmented_sequences.append(sequence)
        augmented_sequences.append(get_random_transposition(sequence))
    return augmented_sequences


def generate_dataset(split_name: str = "train") -> np.ndarray:
    """

    :param split_name:
    :return:
    """
    midi_dir = join(root_dir, "dataset", "mono-midi-transposition-dataset", "midi_files", split_name, "midi")
    print(f"Loading data from path ${midi_dir}.")
    midi_sequences = preprocess_midi(midi_dir)
    print(f"Augmenting dataset.")
    aug_midi_sequences = _augment_dataset(midi_sequences)
    print(f"Padding dataset.")
    padded_sequences = pad(aug_midi_sequences)
    return padded_sequences


def save_dataset(split_name: str):
    """

    :param split_name:
    :return:
    """
    dataset = torch.tensor(generate_dataset(split_name)).to(dtype=torch.int32)
    artifact_path = join(
        root_dir, "dataset", "mono-midi-transposition-dataset", "midi_files", split_name, f"{split_name}.pt"
    )
    torch.save(dataset, artifact_path)


if __name__ == "__main__":
    save_dataset("train")
