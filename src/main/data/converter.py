from typing import List

import numpy as np

BAR_PAD_TOKEN: int = 2
POSITION_PAD_TOKEN: int = 16
PITCH_PAD_TOKEN: int = 86
DURATION_PAD_TOKEN: int = 64
padding_word = [BAR_PAD_TOKEN, POSITION_PAD_TOKEN, PITCH_PAD_TOKEN, DURATION_PAD_TOKEN]


def convert_to_midibert(midi_sequences: List[np.ndarray]) -> np.ndarray:
    """
    Converts the preprocessed MIDI files into the compound words (CPs) in the
    same format as the MidiBERT pre-training data.
    :param midi_sequences: a list of MIDI sequence tuples
    :return: a padded dataset of
    """
    pass
