# Data Specification (V1)

## Processed NPZ format (training input)

Each dataset file is a `.npz` containing:
- `X`: (N, Cin, H, W) float32
- `Y`: (N, Cout, H, W) float32

Optionally:
- `meta`: JSON string, can include channel names, grid size, etc.

## Channel convention (recommended)

Inverse (current phase):
- X channels (12):
  1. Ex_re  2. Ex_im
  3. Ey_re  4. Ey_im
  5. Ez_re  6. Ez_im
  7. Hx_re  8. Hx_im
  9. Hy_re 10. Hy_im
 11. Hz_re 12. Hz_im

- Y channels (4):
  1. Hx_re 2. Hx_im
  3. Hy_re 4. Hy_im

## Where to put datasets

- `NFS_WORKDIR/processed/inverse/train.npz`
- `NFS_WORKDIR/processed/inverse/val.npz`

> Note: V1 does not force a raw CSV schema. If you have legacy CSV/NPZ from earlier scripts,
> add a one-time converter in `src/nfs_ml/data/build_dataset.py` (template provided).
