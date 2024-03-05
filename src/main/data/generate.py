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


def _add_transpositions(midi_sequences: List[np.ndarray], num_transpositions: int = 1) -> List[np.ndarray]:
    """
    Adds a transposition of each input sequence into a new random key
    signature.
    :param midi_sequences: the original MIDI sequences
    :return: all original MIDI sequences, alongside new transpositions
    """
    augmented_sequences = []
    for sequence in tqdm(midi_sequences):
        augmented_sequences.append(sequence)
        for _ in range(num_transpositions):
            augmented_sequences.append(get_random_transposition(sequence))
    return augmented_sequences


def _add_accidentals(midi_sequences: List[np.ndarray], num_accidentals: int = 1) -> List[np.ndarray]:
    """
    Adds a copy of each input sequence with random pitch adjustments.
    :param midi_sequences: the original MIDI sequences
    :return: all original MIDI sequences, alongside new transpositions
    """
    augmented_sequences = []
    for sequence in tqdm(midi_sequences):
        augmented_sequences.append(sequence)
        for _ in range(num_accidentals):
            augmented_sequences.append(add_accidentals(sequence, p=0.1))
    return augmented_sequences


def _shuffle_pairs(midi_sequences: np.ndarray, samples_per_track: int) -> np.ndarray:
    """
    Shuffles all midi sequences belonging to the same original midi file.
    :param midi_sequences: all midi files (including augmented tracks)
    :param samples_per_track: the number of tracks per midi file
    :return: the shuffled dataset
    """
    batched_sequences = midi_sequences.reshape((-1, samples_per_track, *midi_sequences.shape[1:]))
    for i, batch in enumerate(batched_sequences):
        np.random.shuffle(batch)
    return batched_sequences.reshape((-1, *midi_sequences.shape[1:]))


def generate_mono_midi_dataset(split_name: str = "train") -> np.ndarray:
    """
    Generates the mono-midi-transposition-dataset into the MidiBERT format,
    with sequences trimmed to meet size restrictions, and transpositions added
    to augment the dataset.
    :param split_name: the data split (i.e. train, validation, evaluation)
    :return: the dataset represented by a numpy array
    """
    num_transpositions = 1  # should be 1 for validation and evaluation
    num_accidentals = 0  # should be 0 for validation and evaluation
    midi_dir = join(root_dir, "dataset", "mono-midi-transposition-dataset", "midi_files", split_name, "midi")
    print(f"Loading data from path ${midi_dir}.")
    midi_sequences = preprocess_midi(midi_dir)
    print("Splitting dataset.")
    split_midi_sequences = _split_sequences(midi_sequences)
    print("Augmenting dataset.")
    aug_midi_sequences = _add_transpositions(split_midi_sequences, num_transpositions)
    aug_midi_sequences = _add_accidentals(aug_midi_sequences, num_accidentals)
    print("Padding dataset.")
    padded_sequences = pad(aug_midi_sequences)
    print("Shuffling pairs.")
    samples_per_track = (num_transpositions + 1) * (num_accidentals + 1)
    padded_sequences = _shuffle_pairs(padded_sequences, samples_per_track)
    print(len(padded_sequences))
    return padded_sequences


if __name__ == "__main__":
    split = "train"
    dataset = torch.tensor(generate_mono_midi_dataset(split)).to(dtype=torch.int32)
    artifact_path = join(
        root_dir, "dataset", "mono-midi-transposition-dataset", "midi_files", split, f"{split}.pt"
    )
    torch.save(dataset, artifact_path)
