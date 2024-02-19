from musicautobot.config import *
from musicautobot.music_transformer import *
from musicautobot.utils import midifile

# Example File
midi_file = Path('dataset/mono-midi-transposition-dataset/midi_files/train/midi/435.mid')

# Vocab defines how the item is represented as a tensor
vocab = MusicVocab.create()
# Load MusicItem
item = MusicItem.from_file(midi_file, vocab)

# Transpose the item
transposed_item = item.transpose(7)  # Transpose by a perfect fifth (7 semitones)

# Show original and transposed items
print("Original Item:")
item.show()
print("\nTransposed Item:")
transposed_item.show()
