from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

@dataclass
class NpzPair:
    X: np.ndarray  # (N, Cin, H, W)
    Y: np.ndarray  # (N, Cout, H, W)
    meta: dict | None = None

def load_npz_pair(path: Path) -> NpzPair:
    d = np.load(path, allow_pickle=True)
    if "X" not in d or "Y" not in d:
        raise ValueError(f"NPZ must contain keys 'X' and 'Y'. Got keys={list(d.keys())}")
    X = d["X"].astype(np.float32)
    Y = d["Y"].astype(np.float32)
    meta = None
    if "meta" in d:
        meta = d["meta"].item() if hasattr(d["meta"], "item") else d["meta"]
    return NpzPair(X=X, Y=Y, meta=meta)

class FieldNpzDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        assert X.ndim == 4 and Y.ndim == 4, "Expect X,Y shape (N,C,H,W)"
        assert X.shape[0] == Y.shape[0], "X and Y must have same N"
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])  # (Cin,H,W)
        y = torch.from_numpy(self.Y[idx])  # (Cout,H,W)
        return x, y
