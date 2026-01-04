# NFS-ML Suite (Commercial Delivery Layout) — V1

This repository contains **code + configs + docs** only.
All large data, runs, checkpoints, exports, and logs must live in a separate **Workdir**.

## 1) Install (Windows/Linux)

### Option A: editable install
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -U pip
pip install -e .
```

### Option B: install deps only (for quick testing)
```bash
pip install -r requirements.txt
```

## 2) Set Workdir

Create a workdir **outside** this repo (recommended), e.g.:

- Windows: `D:\NFS_WORKDIR`
- Linux: `/data/nfs_workdir`

Set environment variable:

**Windows (PowerShell)**
```powershell
setx NFS_WORKDIR "D:\NFS_WORKDIR"
```

**Linux/macOS**
```bash
export NFS_WORKDIR=/data/nfs_workdir
```

Initialize folder layout:
```bash
python -m nfs_ml init-workdir
```

## 3) Data format (Processed NPZ)

To keep V1 stable and fast, training consumes **processed** `.npz` files:

- `X`: shape `(N, Cin, H, W)`  float32
- `Y`: shape `(N, Cout, H, W)` float32

Example:
- Inverse (your current goal):
  - `X`: 12 channels = (Ex,Ey,Ez,Hx,Hy,Hz) re/im
  - `Y`: 4 channels  = (Hx,Hy) re/im

Place processed datasets here:
- `NFS_WORKDIR/processed/inverse/train.npz`
- `NFS_WORKDIR/processed/inverse/val.npz`

## 4) Train

Inverse example:
```bash
python -m nfs_ml train --task inverse --config configs/model_inverse.yaml
```

Artifacts are saved under:
`NFS_WORKDIR/runs/<run_id>/...`

## 5) Predict + Export

Predict on a processed npz:
```bash
python -m nfs_ml predict --run <run_id> --input processed/inverse/val.npz --out exports/pred_inverse_val.npz
```

Export to CST-style CSV (grid coordinates from config):
```bash
python -m nfs_ml export-csv --pred exports/pred_inverse_val.npz --out exports/pred_inverse_val.csv --config configs/data.yaml
```

## 6) Why PyCharm becomes smooth

This repo intentionally excludes:
- raw data
- processed data
- runs/checkpoints
- exports
- cache

so IDE won't index huge files.

See `docs/05_delivery_checklist.md` for packaging rules.
