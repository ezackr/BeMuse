from typing import List


def preprocess_midi(midi_paths: List[str]):
    """
    Preprocess a list of MIDI files into the CP tuples used for MidiBERT.
    :param midi_paths: MIDI files
    :return: CP tuples corresponding to each MIDI file
    """
    from typing import List
    import pretty_midi

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
                midi_data = pretty_midi.PrettyMIDI(midi_path)

                # Extract musical features (e.g., notes, chords, durations, velocities, etc.)
                # You can use the features you think are relevant for your CP representation

                # Convert musical features into compound words based on vocabulary or encoding
                compound_words = extract_compound_words(midi_data)

                # Append compound words to list
                cp_tuples.append(compound_words)

            except Exception as e:
                print(f"Error processing {midi_path}: {e}")

        return cp_tuples

    def extract_compound_words(midi_data: pretty_midi.PrettyMIDI) -> List[str]:
        """
        Extract compound words from MIDI data.
        This is a placeholder function. You need to implement it based on your specific requirements.
        :param midi_data: MIDI data
        :return: List of compound words
        """
        # Placeholder implementation, replace with your own logic
        # For example, you can extract notes and durations and encode them into compound words
        compound_words = []
        for note in midi_data.instruments[0].notes:
            compound_words.append(f"{note.pitch}_{note.velocity}_{note.start}_{note.end}")
        return compound_words
