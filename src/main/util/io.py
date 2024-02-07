import os
import pickle

import torch
from torch import nn
from transformers import BertConfig

from src.main.model import MidiBert

max_seq_len: int = 512
hidden_dim: int = 768


def get_parent_dir(path: str, level: int = 1):
    """
    Returns the n-th parent directory of a path (where n is given by the
    "level" parameter)
    :param path: the root path
    :param level: the number of parent directories above the root path
    :return:
    """
    if level == 0:
        return path
    if level < 0:
        raise ValueError(f"Parent directory level must be greater than 0. Actual: {level}")
    for _ in range(level):
        path = os.path.dirname(path)
    return path


current_dir: str = os.getcwd()
root_dir: str = get_parent_dir(current_dir, level=3)
dict_path: str = os.path.join(root_dir, "artifact", "midibert", "CP.pkl")
midibert_artifact_path: str = os.path.join(root_dir, "artifact", "midibert", "melody_best.ckpt")


def load_midibert() -> nn.Module:
    # initialize bert model
    configuration = BertConfig(
        max_position_embeddings=max_seq_len,
        position_embedding_type="relative_key_query",
        hidden_size=hidden_dim
    )
    with open(dict_path, "rb") as f:
        e2w, w2e = pickle.load(f)
    model = MidiBert(bertConfig=configuration, e2w=e2w, w2e=w2e)
    # load artifact
    checkpoint = torch.load(midibert_artifact_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    return model
