from __future__ import annotations
from pathlib import Path
import csv
import matplotlib.pyplot as plt

def plot_loss(metrics_csv: Path, out_png: Path) -> None:
    epochs, tr, va = [], [], []
    with open(metrics_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            epochs.append(int(row["epoch"]))
            tr.append(float(row["train_loss"]))
            va.append(float(row["val_loss"]))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(epochs, tr, label="train")
    plt.plot(epochs, va, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
