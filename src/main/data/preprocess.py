import math
from os import walk
from os.path import join
from typing import List, Tuple

import numpy as np
from pretty_midi import PrettyMIDI, TimeSignature
from tqdm import tqdm

NUM_CLASSES: int = 4
BAR_PAD_TOKEN: int = 2
POSITION_PAD_TOKEN: int = 16
PITCH_PAD_TOKEN: int = 86
DURATION_PAD_TOKEN: int = 64
PAD_WORD = [BAR_PAD_TOKEN, POSITION_PAD_TOKEN, PITCH_PAD_TOKEN, DURATION_PAD_TOKEN]

NUM_POSITION_SUB_BEATS: int = 16
NUM_DURATION_SUB_BEATS: int = 16


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


def _classify_bar(prev_bar_number: int, new_bar_number: int) -> int:
    """
    Checks if two given bar numbers are equal.
    :param prev_bar_number: the previous bar number
    :param new_bar_number: the new bar number
    :return: 0 if the bar numbs are the same, 1 otherwise
    """
    return int(prev_bar_number != new_bar_number)


def midi_to_tuple(file_path) -> List[Tuple[int, float, int, float]]:
    """
    Converts a MIDI file into a sequence of 4-tuples, where each tuple has
    the form:
    (bar, position, pitch, duration)
    :param file_path: a MIDI file
    :return: the corresponding sequence of tuples
    """
    try:
        midi_data = PrettyMIDI(file_path)
    except ValueError:
        print(f"Unable to process MIDI file {file_path}")
        return []
    bar_to_ticks = _get_bar_to_ticks_array(midi_data)
    words = []
    prev_bar_number = -1
    for note in midi_data.instruments[0].notes:
        note_start_tick = midi_data.time_to_tick(note.start)
        note_end_tick = midi_data.time_to_tick(note.end)
        bar_number = get_bar_of_tick(note_start_tick, bar_to_ticks)
        position = get_position_of_tick(note_start_tick, bar_number, bar_to_ticks)
        duration = round((note_end_tick - note_start_tick) / midi_data.resolution * NUM_DURATION_SUB_BEATS)
        word = (_classify_bar(prev_bar_number, bar_number), position, note.pitch, duration)
        words.append(word)
        prev_bar_number = bar_number
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
            sequence = np.array(midi_to_tuple(abs_path))
            if len(sequence) != 0:
                midi_sequences.append(sequence)
    return midi_sequences


def pad(midi_sequences: List[np.ndarray], max_length: int = None) -> np.ndarray:
    """
    Pads a given list of midi sequences to a maximum length. If no length is
    provided, the maximum sequence length is used.
    :param midi_sequences: an array of midi sequences
    :param max_length: the length to pad to
    :return: an array corresponding to the padded sequences
    """
    if not max_length:
        max_length = max([len(seq) for seq in midi_sequences])
    padded_seqs = np.zeros(shape=(len(midi_sequences), max_length, NUM_CLASSES))
    for i, seq in enumerate(midi_sequences):
        padding = np.array(PAD_WORD * (max_length - len(seq))).reshape(-1, NUM_CLASSES)
        padded_seqs[i] = np.vstack([seq, padding])
    return padded_seqs
