from typing import List, Tuple
from pretty_midi import PrettyMIDI, Note, TimeSignature


def _time_signature_has_changed(time_sig_changes: List[TimeSignature], note: Note) -> bool:
    """
    Checks if the time signature of a MIDI file has changed.
    :param time_sig_changes: all time signature changes in the MIDI file
    :param note: the current note in the MIDI file
    :return: true iff the current note starts after the first new time
    signature
    """
    return time_sig_changes and note.start >= time_sig_changes[0].time


def midi_to_tuple(file_path) -> List[Tuple[int, int, int, int]]:
    """
    Converts a MIDI file into a sequence of 4-tuples of:
    (bar, position, pitch, duration)
    :param file_path: a MIDI file
    :return: the corresponding sequence of tuples
    """
    midi_data = PrettyMIDI(file_path)
    time_sig_changes = midi_data.time_signature_changes
    current_bar_number = 1
    current_time_sig = (4, 4)
    words = []
    for note in midi_data.instruments[0].notes:
        while _time_signature_has_changed(time_sig_changes, note):
            current_time_sig = (time_sig_changes[0].numerator, time_sig_changes[0].denominator)
            time_sig_changes.pop(0)
            current_bar_number = 1
        bar_number = current_bar_number + int((note.start / midi_data.get_end_time()) * current_time_sig[0])
        position = note.start % (current_time_sig[0] / current_time_sig[1])
        word = (bar_number, position, note.pitch, note.end - note.start)
        words.append(word)
    return words


def preprocess_midi(midi_paths: List[str]) -> List[List[Tuple[int, int, int, int]]]:
    """
    Preprocess a list of MIDI files into the CP tuples used for MidiBERT. This
    converts a directory of MIDI files into a corresponding list of compound words.
    :param midi_paths: MIDI files
    :return: CP tuples corresponding to each MIDI file
    """
    return [midi_to_tuple(midi_path) for midi_path in midi_paths]
