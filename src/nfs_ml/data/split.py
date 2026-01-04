from __future__ import annotations
import numpy as np

def split_indices(n: int, val_ratio: float = 0.1, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(round(n * val_ratio))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx
