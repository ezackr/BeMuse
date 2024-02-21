from typing import Tuple

import torch
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.main.model import MidiBert
from src.main.util import load_midibert, load_mono_midi_trans_dataset, pairwise_loss, save_midibert

NUM_EPOCHS: int = 4
BATCH_SIZE: int = 16


def get_dataloaders() -> Tuple[DataLoader, DataLoader]:
    train_tensors = load_mono_midi_trans_dataset("train")
    train_tensors = train_tensors.view(-1, 2, *train_tensors.size()[1:])
    val_tensors = load_mono_midi_trans_dataset("validation")
    val_tensors = val_tensors.view(-1, 2, *val_tensors.size()[1:])
    train_loader = DataLoader(TensorDataset(train_tensors), batch_size=16, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensors), batch_size=16, shuffle=True)
    return train_loader, val_loader


def train(model: MidiBert, train_loader: DataLoader, val_loader: DataLoader, optimizer: Optimizer):
    train_history = []
    val_history = []
    for _ in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader):
            original = batch[0][:, 0, :, :]
            transpose = batch[0][:, 1, :, :]

            optimizer.zero_grad()

            original_vec = model(original)
            transpose_vec = model(transpose)

            loss = pairwise_loss(original_vec, transpose_vec)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()
        train_history.append(train_loss / len(train_loader))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader):
                original = batch[0][:, 0, :, :]
                transpose = batch[0][:, 1, :, :]
                original_vec = model(original)
                transpose_vec = model(transpose)
                loss = pairwise_loss(original_vec, transpose_vec)
                val_loss += loss.item()
        val_history.append(val_loss / len(val_loader))

        print(f"Epoch {len(train_history)}, train-loss={train_history[-1]}, val-loss={val_history[-1]}")
        save_midibert(model, f"midibert-epoch{len(train_history)}")
    return train_history, val_history


def main():
    model = load_midibert()
    train_loader, val_loader = get_dataloaders()
    optimizer = Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    train(model, train_loader, val_loader, optimizer)


if __name__ == "__main__":
    main()
