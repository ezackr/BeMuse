from os import walk
from os.path import join
from typing import List, Tuple
from pretty_midi import PrettyMIDI, TimeSignature
from tqdm import tqdm


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
    bar_ticks = []
    while current_tick < max_tick:
        while _time_signature_has_changed(time_sig_changes, current_tick):
            current_time_sig = (time_sig_changes[0].numerator, time_sig_changes[0].denominator)
            time_sig_changes.pop(0)
        num_quarter_notes_per_bar = (current_time_sig[0] / current_time_sig[1]) * 4
        current_tick += num_quarter_notes_per_bar * midi_data.resolution
        bar_ticks.append(current_tick)
    return bar_ticks


def _get_bar_of_tick(current_ticks: float, bar_to_ticks: List[float]) -> int:
    """
    Gets the bar of a given tick value. The array bar_to_ticks corresponds to
    the tick value at the start of each bar. The bar of an arbitrary tick
    value corresponds to the largest index in bar_to_ticks such that the
    tick value at the start of the bar is less than or equal to the given tick
    value.
    :param current_ticks: a given tick value
    :param bar_to_ticks: the tick values at the start of each bar
    :return: the bar corresponding to the current_ticks value
    """
    for i in range(len(bar_to_ticks)):
        if bar_to_ticks[i] > current_ticks:
            return i + 1
    return len(bar_to_ticks)


def midi_to_tuple(file_path) -> List[Tuple[int, float, int, float]]:
    """
    Converts a MIDI file into a sequence of 4-tuples of:
    (bar, position, pitch, duration)
    :param file_path: a MIDI file
    :return: the corresponding sequence of tuples
    """
    midi_data = PrettyMIDI(file_path)
    bar_to_ticks = _get_bar_to_ticks_array(midi_data)
    words = []
    for note in midi_data.instruments[0].notes:
        bar_number = _get_bar_of_tick(midi_data.time_to_tick(note.start), bar_to_ticks)
        position = -1
        word = (bar_number, position, note.pitch, note.end - note.start)
        words.append(word)
    return words


def preprocess_midi(midi_dir: str) -> List[List[Tuple[int, float, int, float]]]:
    """
    Preprocess a list of MIDI files into the CP tuples used for MidiBERT. This
    converts a directory of MIDI files into a corresponding list of compound words.
    :param midi_dir: A directory of midi files
    :return: CP tuples corresponding to each MIDI file
    """
    for root, _, files in walk(midi_dir):
        for file in tqdm(files):
            file_path = join(root, file)
            midi_to_tuple(file_path)
    return []
