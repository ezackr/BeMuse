import os
import pickle

import torch
from transformers import BertConfig

from src.main.model import MidiBert

max_seq_len: int = 512
hidden_dim: int = 768


def get_parent_dir(path: str, level: int = 1) -> str:
    """
    Returns the (level)-th parent directory of a given path
    :param path: the root path
    :param level: the number of parent directories above the root path
    :return: the (level)-th parent directory of the given path
    :raise ValueError: if the given level is less than 0
    """
    if level == 0:
        return path
    if level < 0:
        raise ValueError(f"Parent directory level must be greater than 0. Actual: {level}")
    for _ in range(level):
        path = os.path.dirname(path)
    return path


current_path: str = os.path.abspath(__file__)
root_dir: str = get_parent_dir(current_path, level=4)
dict_path: str = os.path.join(root_dir, "artifact", "midibert", "CP.pkl")


def load_midibert(artifact_name: str = "pretrain_model.ckpt") -> MidiBert:
    """
    Loads the pre-trained MidiBERT checkpoint from the artifact directory.
    :param artifact_name: the name of the MidiBERT artifact file in the
    "BeMuse/artifact/midibert/" directory
    :return: the pre-trained melody MidiBERT encoder
    :raise ValueError: if the MidiBERT checkpoint does not exist in the
    artifact directory
    """
    # check if checkpoint exists
    midibert_artifact_path = os.path.join(root_dir, "artifact", "midibert", artifact_name)
    if not os.path.exists(midibert_artifact_path):
        raise ValueError("Unable to find artifact file " + midibert_artifact_path)
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
    state_dict = torch.load(midibert_artifact_path, map_location="cpu")["state_dict"]
    del state_dict["bert.embeddings.position_ids"]
    model.load_state_dict(state_dict=state_dict)
    return model


def save_midibert(model: MidiBert, artifact_name: str):
    """
    Saves a MidiBERT model state into a given artifact file.
    :param model: the MidiBERT model
    :param artifact_name: the name of the artifact
    """
    midibert_artifact_path = os.path.join(root_dir, "artifact", "midibert", artifact_name)
    torch.save(model.state_dict, midibert_artifact_path)


def load_mono_midi_trans_dataset(split_name: str = "train") -> torch.Tensor:
    """
    Loads a split of the mono-midi-transposition-dataset into a PyTorch
    tensor.
    :param split_name: the data split (i.e. train, validation, evaluation)
    :return: the dataset as a PyTorch tensor
    """
    dataset_path = os.path.join(
        root_dir, "dataset", "mono-midi-transposition-dataset",
        "midi_files", split_name, f"{split_name}.pt"
    )
    return torch.load(dataset_path)
