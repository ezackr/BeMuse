import os

from src.main.data.preprocess import get_bar_of_tick, get_position_of_tick, midi_to_tuple
from src.main.util.io import root_dir


def test_get_bar_of_tick():
    bar_to_ticks = [0.0, 20.0, 40.0]
    assert get_bar_of_tick(0.0, bar_to_ticks) == 1
    assert get_bar_of_tick(15.0, bar_to_ticks) == 1
    assert get_bar_of_tick(20.0, bar_to_ticks) == 2
    assert get_bar_of_tick(40.0, bar_to_ticks) == 3


def test_get_position_of_tick():
    bar_to_ticks = [0.0, 20.0, 40.0]
    assert get_position_of_tick(0.0, 1, bar_to_ticks) == 0
    assert get_position_of_tick(4.0, 1, bar_to_ticks) == 0.1875
    assert get_position_of_tick(10.0, 1, bar_to_ticks) == 0.5


def test_midi_to_tuple():
    example_path = os.path.join(
        root_dir, "dataset", "mono-midi-transposition-dataset", "midi_files", "train", "midi", "435.mid"
    )
    sequence = midi_to_tuple(example_path)
    for i in range(5):
        assert sequence[i][0] == 1
