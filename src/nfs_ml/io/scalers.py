from __future__ import annotations
import numpy as np

EPS = 1e-8

def fit_channel_zscore(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """arr: (N, C, H, W)"""
    mean = arr.mean(axis=(0,2,3), keepdims=True)
    std  = arr.std(axis=(0,2,3), keepdims=True) + EPS
    return mean.astype(np.float32), std.astype(np.float32)

def apply_zscore(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((arr - mean) / std).astype(np.float32)

def save_scaler(path, x_mean, x_std, y_mean, y_std, meta: dict | None = None) -> None:
    payload = dict(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
    if meta is not None:
        payload["meta"] = np.array([np.string_(str(meta))])
    np.savez(path, **payload)

def load_scaler(path):
    s = np.load(path, allow_pickle=True)
    meta = None
    if "meta" in s:
        try:
            meta = s["meta"][0].decode("utf-8")
        except Exception:
            meta = str(s["meta"][0])
    return s["x_mean"], s["x_std"], s["y_mean"], s["y_std"], meta
