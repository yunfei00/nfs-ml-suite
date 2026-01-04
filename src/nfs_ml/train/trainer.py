from __future__ import annotations
import csv
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "auto"   # auto|cpu|cuda
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    grad_clip: float = 1.0
    loss: str = "smoothl1"

def pick_device(device: str) -> torch.device:
    device = device.lower().strip()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_loop(
    model: nn.Module,
    loss_fn: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    out_dir: Path,
    cfg: TrainConfig,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    plot_dir = out_dir / "plots"
    log_dir = out_dir / "logs"
    ckpt_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    device = pick_device(cfg.device)
    set_seed(cfg.seed)

    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    metrics_path = out_dir / "metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "seconds"])

    best_val = float("inf")
    for ep in range(1, cfg.epochs + 1):
        t0 = time.time()
        model.train()
        tr = 0.0
        ntr = 0

        for x, y in tqdm(train_loader, desc=f"epoch {ep:03d} train", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            bs = x.size(0)
            tr += loss.item() * bs
            ntr += bs

        tr_loss = tr / max(1, ntr)

        model.eval()
        va = 0.0
        nva = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"epoch {ep:03d} val", leave=False):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                pred = model(x)
                loss = loss_fn(pred, y)
                bs = x.size(0)
                va += loss.item() * bs
                nva += bs

        va_loss = va / max(1, nva)
        dt = time.time() - t0

        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([ep, f"{tr_loss:.8f}", f"{va_loss:.8f}", f"{dt:.2f}"])

        print(f"[epoch {ep:03d}] train={tr_loss:.6f} val={va_loss:.6f} sec={dt:.1f}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save({"model": model.state_dict(), "epoch": ep, "val_loss": va_loss}, ckpt_dir / "best.pt")
            print(f"  saved: {ckpt_dir / 'best.pt'}")

    return metrics_path
