from typing import List

from torch import nn

from src.main.data import preprocess_midi
from src.main.util import load_midibert


class BeMuse(nn.Module):
    def __init__(self, midibert_ckpt: str):
        super(BeMuse, self).__init__()
        self.midibert = load_midibert(midibert_ckpt)

    def forward(self, midi_file: str) -> List[str]:
        cp = preprocess_midi([midi_file])
        enc = self.midibert(cp)
        return []
