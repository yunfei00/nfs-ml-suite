# Training Guide (V1)

## Quick start (inverse)

1) Ensure processed datasets exist:
- `processed/inverse/train.npz`
- `processed/inverse/val.npz`

2) Train:
```bash
python -m nfs_ml train --task inverse --config configs/model_inverse.yaml
```

Outputs:
- `runs/<run_id>/checkpoints/best.pt`
- `runs/<run_id>/scaler.npz`
- `runs/<run_id>/metrics.csv`
- `runs/<run_id>/plots/loss.png`

## Reproducibility
- Every run snapshots a resolved config: `runs/<run_id>/config_resolved.yaml`.
- Keep run folder for audit; delete `cache/` anytime.
