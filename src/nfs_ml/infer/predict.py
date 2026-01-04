from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from nfs_ml.io.dataset_npz import load_npz_pair, FieldNpzDataset
from nfs_ml.io.scalers import load_scaler, apply_zscore
from nfs_ml.models.registry import build_model

def predict_npz(
    input_npz: Path,
    scaler_npz: Path,
    ckpt_pt: Path,
    model_cfg: dict,
    out_npz: Path,
    device: str = "auto",
):
    pair = load_npz_pair(input_npz)

    x_mean, x_std, y_mean, y_std, _ = load_scaler(scaler_npz)
    Xn = apply_zscore(pair.X, x_mean, x_std)

    ds = FieldNpzDataset(Xn, pair.Y)  # Y only for shape; not used
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

    Cin, H, W = pair.X.shape[1], pair.X.shape[2], pair.X.shape[3]
    Cout = pair.Y.shape[1]
    model = build_model(model_cfg["name"], cin=Cin, cout=Cout, H=H, W=W, cfg=model_cfg)

    dev = torch.device("cuda" if (device=="auto" and torch.cuda.is_available()) else ("cpu" if device=="auto" else device))
    model.to(dev)
    state = torch.load(ckpt_pt, map_location=dev)
    model.load_state_dict(state["model"])
    model.eval()

    preds = []
    with torch.no_grad():
        for x, _ in tqdm(dl, desc="predict"):
            x = x.to(dev)
            y = model(x).cpu().numpy()
            preds.append(y)

    Ypred_n = np.concatenate(preds, axis=0).astype(np.float32)
    # de-normalize
    Ypred = (Ypred_n * y_std + y_mean).astype(np.float32)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, Y_pred=Ypred, meta=np.array({"source": str(input_npz)}, dtype=object))
    return out_npz
