import pickle
from os.path import join

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import BertConfig

from src.main.model import MidiBert
from src.main.util import load_mono_midi_trans_dataset, root_dir

BATCH_SIZE: int = 16


def load_model(artifact_name: str = "midibert-ckpt-10") -> MidiBert:
    midibert_artifact_path = join(root_dir, "artifact", "midibert", artifact_name)
    configuration = BertConfig(
        max_position_embeddings=512,
        position_embedding_type="relative_key_query",
        hidden_size=768
    )
    dict_path = join(root_dir, "artifact", "midibert", "CP.pkl")
    with open(dict_path, "rb") as f:
        e2w, w2e = pickle.load(f)
    model = MidiBert(bertConfig=configuration, e2w=e2w, w2e=w2e)
    state_dict = torch.load(midibert_artifact_path, map_location="cpu")
    model.load_state_dict(state_dict=state_dict)
    return model


def get_dataloaders() -> DataLoader:
    eval_tensors = load_mono_midi_trans_dataset("train")
    eval_tensors = eval_tensors.view(-1, 2, *eval_tensors.size()[1:])
    eval_loader = DataLoader(TensorDataset(eval_tensors), batch_size=BATCH_SIZE, shuffle=False)
    return eval_loader


def get_similarity(queries: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    queries_norm = F.normalize(queries, p=2, dim=1)
    targets_norm = F.normalize(targets, p=2, dim=1)
    return torch.matmul(targets_norm, queries_norm.T)


def compute_accuracy(top_k: torch.Tensor):
    accuracy = 0
    for i, sim in enumerate(top_k):
        if i in sim:
            accuracy += 1
    return accuracy / len(top_k)


def evaluate(model: MidiBert, eval_loader: DataLoader, k: int = 5):
    model.to(device)
    model.eval()
    enc_queries = []
    enc_targets = []
    with torch.no_grad():
        for batch in tqdm(eval_loader):
            queries = batch[0][:, 1, :, :].to(device)
            targets = batch[0][:, 0, :, :].to(device)
            queries_vec = model(queries)
            targets_vec = model(targets)
            enc_queries += [q for q in queries_vec]
            enc_targets += [t for t in targets_vec]
    enc_queries = torch.vstack(enc_queries)
    enc_targets = torch.vstack(enc_targets)
    similarity_matrix = get_similarity(enc_queries, enc_targets)
    top_k = torch.argsort(similarity_matrix, dim=1, descending=True)[:, :k]
    return compute_accuracy(top_k)


def main():
    k = 5
    artifact_name = "midibert-ckpt-10"
    model = load_model(artifact_name)
    eval_loader = get_dataloaders()
    accuracy = evaluate(model, eval_loader, k)
    print(f"Top {k} accuracy = {accuracy}")


if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    main()
