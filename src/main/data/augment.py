import numpy as np

MIN_MIDI_PITCH: int = 0
MAX_MIDI_PITCH: int = 127


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
    trans_midi_sequence[:, 2] = np.clip(trans_midi_sequence[:, 2], MIN_MIDI_PITCH, MAX_MIDI_PITCH)
    return trans_midi_sequence
