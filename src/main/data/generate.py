from os.path import join
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from src.main.data import add_accidentals, get_random_transposition, pad, preprocess_midi, split_to_length
from src.main.util import root_dir

MAX_BERT_SEQ_LENGTH: int = 512


def _split_sequences(midi_sequences: List[np.ndarray]) -> List[np.ndarray]:
    """
    Splits larger MIDI sequences into multiple MIDI sequences of a fixed
    maximum length.
    :param midi_sequences: the original MIDI sequences
    :return: the shortened, augmented MIDI sequences
    """
    split_sequences = []
    for sequence in tqdm(midi_sequences):
        split_sequences += split_to_length(sequence, MAX_BERT_SEQ_LENGTH)
    return split_sequences


def _add_transpositions(midi_sequences: List[np.ndarray]) -> List[np.ndarray]:
    """
    Adds a transposition of each input sequence into a new random key
    signature.
    :param midi_sequences: the original MIDI sequences
    :return: all original MIDI sequences, alongside new transpositions
    """
    augmented_sequences = []
    for sequence in tqdm(midi_sequences):
        augmented_sequences.append(sequence)
        augmented_sequences.append(get_random_transposition(sequence))
    return augmented_sequences


def _add_accidentals(midi_sequences: List[np.ndarray]) -> List[np.ndarray]:
    """
    Adds a copy of each input sequence with random pitch adjustments.
    :param midi_sequences: the original MIDI sequences
    :return: all original MIDI sequences, alongside new transpositions
    """
    augmented_sequences = []
    for sequence in tqdm(midi_sequences):
        augmented_sequences.append(sequence)
        augmented_sequences.append(add_accidentals(sequence, p=0.1))
    return augmented_sequences


def generate_mono_midi_dataset(split_name: str = "train") -> np.ndarray:
    """
    Generates the mono-midi-transposition-dataset into the MidiBERT format,
    with sequences trimmed to meet size restrictions, and transpositions added
    to augment the dataset.
    :param split_name: the data split (i.e. train, validation, evaluation)
    :return: the dataset represented by a numpy array
    """
    midi_dir = join(root_dir, "dataset", "mono-midi-transposition-dataset", "midi_files", split_name, "midi")
    print(f"Loading data from path ${midi_dir}.")
    midi_sequences = preprocess_midi(midi_dir)
    print(f"Splitting dataset.")
    split_midi_sequences = _split_sequences(midi_sequences)
    print(f"Augmenting dataset.")
    aug_midi_sequences = _add_transpositions(split_midi_sequences)
    aug_midi_sequences = _add_accidentals(aug_midi_sequences)
    print(f"Padding dataset.")
    padded_sequences = pad(aug_midi_sequences)
    return padded_sequences


if __name__ == "__main__":
    split = "validation"
    dataset = torch.tensor(generate_mono_midi_dataset(split)).to(dtype=torch.int32)
    artifact_path = join(
        root_dir, "dataset", "mono-midi-transposition-dataset", "midi_files", split, f"{split}.pt"
    )
    torch.save(dataset, artifact_path)
