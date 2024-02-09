from typing import List

from torch import nn


class BeMuse(nn.Module):
    def __init__(self, midibert_ckpt: str):
        super(BeMuse, self).__init__()

    def forward(self, midi_file: str) -> List[str]:
        pass

