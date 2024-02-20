from os.path import join

import numpy as np
import torch

from src.main.data import pad, midi_to_tuple
from src.main.util import load_midibert, root_dir


def test_midibert_forward():
    example_seq = torch.tensor(pad([
        np.array(
            [(1, 0, 59, 8), (0, 4, 59, 8), (0, 8, 59, 8), (0, 12, 59, 8), (0, 14, 59, 8), (1, 0, 59, 8), (0, 2, 59, 8),
             (0, 4, 59, 8)]),
        np.array([(1, 0, 59, 8), (0, 14, 59, 8), (1, 0, 59, 8), (0, 2, 59, 8), (0, 4, 59, 8)]),
        np.array([(1, 0, 59, 8), (0, 4, 59, 8), (0, 8, 59, 8), (0, 12, 59, 8), (0, 14, 59, 8), (1, 0, 59, 8)])
    ])).to(dtype=torch.int32)
    model = load_midibert()
    assert model(example_seq).shape == (3, 8, 768)


def test_midibert_forward_midi():
    path = join(root_dir, "dataset", "mono-midi-transposition-dataset", "midi_files", "train", "midi")
    first_seq = np.array(midi_to_tuple(join(path, "435.mid")))
    second_seq = np.array(midi_to_tuple(join(path, "524.mid")))
    example_seq = torch.tensor(pad([first_seq, second_seq])).to(dtype=torch.int32)
    model = load_midibert()
    print(model(example_seq))
