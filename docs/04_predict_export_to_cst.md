# Predict & Export to CST

## Predict
```bash
python -m nfs_ml predict --run <run_id> --input processed/inverse/val.npz --out exports/pred_val.npz
```

## Export CSV (CST-style)
```bash
python -m nfs_ml export-csv --pred exports/pred_val.npz --out exports/pred_val.csv --config configs/data.yaml
```

CSV columns follow config (default):
- x, y, z, hx_re, hx_im, hy_re, hy_im

Coordinates are generated from `configs/data.yaml`:
- x/y range (mm)
- grid H/W
- z fixed (mm)
