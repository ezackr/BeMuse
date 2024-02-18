from typing import List
from pretty_midi import PrettyMIDI


def midi_to_tuple(file_path):
    midi_data = PrettyMIDI(file_path)
    time_signature_changes = midi_data.time_signature_changes
    words = []
    current_bar_number = 1
    current_time_signature = (4, 4)
    for note in midi_data.instruments[0].notes:
        # Check if time signature changes have occurred
        while time_signature_changes and note.start >= time_signature_changes[0].time:
            current_time_signature = (time_signature_changes[0].numerator, time_signature_changes[0].denominator)
            time_signature_changes.pop(0)
            current_bar_number = 1
        # Calculate the bar number for the note
        bar_number = current_bar_number + int((note.start / midi_data.get_end_time()) * current_time_signature[0])
        # Calculate the position within the bar
        position = note.start % (current_time_signature[0] / current_time_signature[1])
        # Add the note information to the list
        note_info = (bar_number, position, note.pitch, note.end - note.start)
        words.append(note_info)
    return words


def _extract_compound_words(midi_data: PrettyMIDI) -> List[str]:
    """
    Extract compound words from MIDI data.
    This is a placeholder function. You need to implement it based on your specific requirements.
    :param midi_data: MIDI data
    :return: List of compound words
    """
    compound_words = []
    for note in midi_data.instruments[0].notes:
        compound_words.append(f"{note.pitch}_{note.velocity}_{note.start}_{note.end}")
    return compound_words


def preprocess_midi(midi_paths: List[str]) -> List[List[str]]:
    """
    Preprocess a list of MIDI files into the CP tuples used for MidiBERT. This
    converts a directory of MIDI files into a corresponding list of compound words.
    :param midi_paths: MIDI files
    :return: CP tuples corresponding to each MIDI file
    """
    cp_tuples = []
    # Iterate over each MIDI file path
    for midi_path in midi_paths:
        try:
            # Load MIDI file
            midi_data = PrettyMIDI(midi_path)
            # Extract musical features (e.g., notes, chords, durations, velocities, etc.)
            # Convert musical features into compound words based on vocabulary or encoding
            compound_words = _extract_compound_words(midi_data)
            # Append compound words to list
            cp_tuples.append(compound_words)
        except Exception as e:
            print(f"Error processing {midi_path}: {e}")
    return cp_tuples
