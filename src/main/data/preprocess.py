import math
from os import walk
from os.path import join
from typing import List, Tuple

import numpy as np
from pretty_midi import PrettyMIDI, TimeSignature
from tqdm import tqdm

BAR_PAD_TOKEN: int = 2
POSITION_PAD_TOKEN: int = 16
PITCH_PAD_TOKEN: int = 86
DURATION_PAD_TOKEN: int = 64
padding_word = [BAR_PAD_TOKEN, POSITION_PAD_TOKEN, PITCH_PAD_TOKEN, DURATION_PAD_TOKEN]

NUM_POSITION_SUB_BEATS: int = 16
NUM_DURATION_SUB_BEATS: int = 64


def _time_signature_has_changed(time_sig_changes: List[TimeSignature], current_time: float) -> bool:
    """
    Checks if the time signature of a MIDI file has changed.
    :param time_sig_changes: all time signature changes in the MIDI file
    :param current_time: the current time in the MIDI file
    :return: true iff the current time passes the first time signature change
    """
    return time_sig_changes and current_time >= time_sig_changes[0].time


def _get_bar_to_ticks_array(midi_data: PrettyMIDI) -> List[float]:
    """
    Creates an array where the i-th entry corresponds to the tick value at the
    start of the i-th bar
    :param midi_data: a MIDI file
    :return: the number of ticks at the start of each bar
    """
    current_tick = 0
    max_tick = midi_data.time_to_tick(midi_data.get_end_time())
    current_time_sig = (4, 4)
    time_sig_changes = midi_data.time_signature_changes
    bar_ticks = [0]
    while current_tick < max_tick:
        while _time_signature_has_changed(time_sig_changes, current_tick):
            current_time_sig = (time_sig_changes[0].numerator, time_sig_changes[0].denominator)
            time_sig_changes.pop(0)
        num_quarter_notes_per_bar = (current_time_sig[0] / current_time_sig[1]) * 4
        current_tick += num_quarter_notes_per_bar * midi_data.resolution
        bar_ticks.append(current_tick)
    return bar_ticks


def get_bar_of_tick(current_tick: float, bar_to_ticks: List[float]) -> int:
    """
    Gets the bar of a given tick value. The array bar_to_ticks corresponds to
    the tick value at the start of each bar. The bar of an arbitrary tick
    value corresponds to the largest index in bar_to_ticks such that the
    tick value at the start of the bar is less than or equal to the given tick
    value.
    :param current_tick: a given tick value
    :param bar_to_ticks: the tick values at the start of each bar
    :return: the bar corresponding to the current_ticks value
    """
    for i in range(len(bar_to_ticks)):
        if bar_to_ticks[i] > current_tick:
            return i
    return len(bar_to_ticks)


def get_position_of_tick(current_tick: float, bar_number: int, bar_to_ticks: List[float]) -> int:
    """
    Gets the position of a given tick value relative to the start of its bar.
    :param current_tick: a given tick value
    :param bar_number: the bar number of the given tick value
    :param bar_to_ticks: the tick values at the start of each bar
    :return: the position of a tick relative to the start of its bar
    """
    bar_length = bar_to_ticks[bar_number] - bar_to_ticks[bar_number - 1]
    abs_pos = (current_tick - bar_to_ticks[bar_number - 1]) / bar_length
    return math.floor(abs_pos * NUM_POSITION_SUB_BEATS)


def get_duration_of_note(start_tick: float, end_tick: float, bar_to_ticks: List[float]) -> int:
    start_bar = get_bar_of_tick(start_tick, bar_to_ticks)
    end_bar = get_bar_of_tick(end_tick, bar_to_ticks)
    return -1


def _classify_bar(bar_number: int, words: List[Tuple[int, float, int, float]]) -> int:
    """
    Checks if the given bar number is new in the given word sequence.
    :param bar_number: the given bar number
    :param words: a MIDI sequence represented by compound words
    :return: 0 if a bar is new, 1 otherwise
    """
    if not words:
        return 0
    return int(bar_number == words[-1])


def midi_to_tuple(file_path) -> List[Tuple[int, float, int, float]]:
    """
    Converts a MIDI file into a sequence of 4-tuples, where each tuple has
    the form:
    (bar, position, pitch, duration)
    :param file_path: a MIDI file
    :return: the corresponding sequence of tuples
    """
    midi_data = PrettyMIDI(file_path)
    bar_to_ticks = _get_bar_to_ticks_array(midi_data)
    words = []
    for note in midi_data.instruments[0].notes:
        note_start_tick = midi_data.time_to_tick(note.start)
        note_end_tick = midi_data.time_to_tick(note.end)
        bar_number = get_bar_of_tick(note_start_tick, bar_to_ticks)
        position = get_position_of_tick(note_start_tick, bar_number, bar_to_ticks)
        duration = get_duration_of_note(note_start_tick, note_end_tick, bar_to_ticks)
        word = (_classify_bar(bar_number, words), position, note.pitch, duration)
        words.append(word)
    return words


def preprocess_midi(midi_dir: str) -> List[np.ndarray]:
    """
    Preprocesses a directory of MIDI files into tuples used for MidiBERT.
    :param midi_dir: a directory of MIDI files
    :return: tuple sequences corresponding to each MIDI file
    """
    midi_sequences = []
    for root, _, files in walk(midi_dir):
        for file in tqdm(files):
            abs_path = join(root, file)
            midi_sequences.append(np.array(midi_to_tuple(abs_path)))
    return midi_sequences
