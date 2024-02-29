from typing import List
import numpy as np

MIN_MIDI_PITCH: int = 0
MAX_MIDI_PITCH: int = 85
MAX_BERT_SEQ_LEN: int = 512


def get_random_transposition(midi_sequence: np.ndarray) -> np.ndarray:
    """
    Creates a random transposition of an original midi sequence by shifting
    each pitch by a random, constant value.
    :param midi_sequence: the original midi sequence
    :return: the new, transposed midi sequence
    """
    transposition_value = np.random.randint(1, 12)
    trans_midi_sequence = midi_sequence.copy()
    trans_midi_sequence[:, 2] += transposition_value
    if any(trans_midi_sequence[:, 2] >= MAX_MIDI_PITCH):
        # attempt to shift pitches down an octave if exceeding maximum value
        trans_midi_sequence[:, 2] -= 12
    trans_midi_sequence[:, 2] = np.clip(trans_midi_sequence[:, 2], MIN_MIDI_PITCH, MAX_MIDI_PITCH)
    return trans_midi_sequence


def add_accidentals(midi_sequence: np.ndarray, p: float = 0.05) -> np.ndarray:
    """
    Adds accidentals to a midi sequence by randomly shifting a note [-2, +2]
    semitones. If the shift creates an invalid midi pitch, then it is not
    shifted.
    :param midi_sequence: the original midi sequence
    :param p: the probability of a note being shifted
    :return: the new midi sequence with added accidentals
    """
    acc_midi_sequence = midi_sequence.copy()
    for i, note in enumerate(midi_sequence):
        if np.random.rand() < p:
            shift = np.random.randint(-2, 2 + 1)
            if MIN_MIDI_PITCH <= note + shift <= MAX_MIDI_PITCH:
                acc_midi_sequence[i, 2] += shift
    return acc_midi_sequence


def _find_next_new_bar(midi_sequence: np.ndarray, start_idx: int) -> int:
    """
    Finds the next recent new bar starting from a given index in a sequence.
    A new bar is characterized by a "1" in the first feature.
    :param midi_sequence: a MIDI sequence
    :param start_idx: the index to begin the search
    :return: the index of the next new bar after the star index. Returns -1 if
    no such index exists.
    """
    for i in range(start_idx, len(midi_sequence)):
        if midi_sequence[i][0] == 1:
            return i
    return -1


def split_to_length(midi_sequence: np.ndarray, max_length: int = MAX_BERT_SEQ_LEN) -> List[np.ndarray]:
    """
    Splits a given sequence into multiple sequences of at most a given size. A
    sequence must begin with a new bar word, which is identified by a "1".
    :param midi_sequence: the original sequence
    :param max_length: the maximum subsequence length
    :return: a list of subsequences within the maximum size
    """
    if len(midi_sequence) <= max_length:
        return [midi_sequence]
    subsequences = [midi_sequence[:max_length]]
    start_idx = max_length
    while any(midi_sequence[start_idx:, 0] == 1):
        next_new_bar_idx = _find_next_new_bar(midi_sequence, start_idx)
        if len(midi_sequence[next_new_bar_idx:]) < max_length:
            subsequences.append(midi_sequence[next_new_bar_idx:])
            start_idx = len(midi_sequence)
        else:
            subsequences.append(midi_sequence[next_new_bar_idx:next_new_bar_idx + max_length])
            start_idx = next_new_bar_idx + max_length
    return subsequences
