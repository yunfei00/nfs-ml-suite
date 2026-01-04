from __future__ import annotations
from pathlib import Path
import numpy as np
import csv
import yaml

def _linspace(a: float, b: float, n: int):
    if n <= 1:
        return np.array([a], dtype=np.float32)
    return np.linspace(a, b, n, dtype=np.float32)

def export_inverse_csv(pred_npz: Path, out_csv: Path, data_cfg_yaml: Path) -> Path:
    cfg = yaml.safe_load(data_cfg_yaml.read_text(encoding="utf-8"))
    grid = cfg["grid"]
    H = int(grid["H"]); W = int(grid["W"])
    x = _linspace(float(grid["x_min_mm"]), float(grid["x_max_mm"]), W)
    y = _linspace(float(grid["y_min_mm"]), float(grid["y_max_mm"]), H)
    z = float(grid.get("z_mm", 1.0))

    d = np.load(pred_npz, allow_pickle=True)
    if "Y_pred" not in d:
        raise ValueError("pred npz must contain 'Y_pred' of shape (N,4,H,W) for inverse export.")
    Y = d["Y_pred"]  # (N,4,H,W)
    if Y.ndim != 4 or Y.shape[1] < 4:
        raise ValueError(f"Unexpected Y_pred shape: {Y.shape}")
    # export first sample by default
    yy = Y[0]  # (4,H,W): hx_re, hx_im, hy_re, hy_im
    hx_re, hx_im, hy_re, hy_im = yy[0], yy[1], yy[2], yy[3]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = cfg.get("export", {}).get("inverse_csv_columns", ["x","y","z","hx_re","hx_im","hy_re","hy_im"])

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for iy in range(H):
            for ix in range(W):
                row = [x[ix], y[iy], z, hx_re[iy,ix], hx_im[iy,ix], hy_re[iy,ix], hy_im[iy,ix]]
                w.writerow(row)

    return out_csv
