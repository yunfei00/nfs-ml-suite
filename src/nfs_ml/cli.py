from __future__ import annotations
import argparse
import datetime as _dt
from pathlib import Path
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from nfs_ml.io.paths import get_workdir, ensure_workdir_layout, join_workdir
from nfs_ml.io.dataset_npz import load_npz_pair, FieldNpzDataset
from nfs_ml.io.scalers import fit_channel_zscore, apply_zscore, save_scaler
from nfs_ml.models.registry import build_model
from nfs_ml.train.losses import build_loss
from nfs_ml.train.trainer import TrainConfig, train_loop
from nfs_ml.viz.plots import plot_loss
from nfs_ml.infer.predict import predict_npz
from nfs_ml.infer.export_cst import export_inverse_csv

def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))

def _now_run_id(prefix: str) -> str:
    return _dt.datetime.now().strftime(f"%Y%m%d-%H%M%S_{prefix}")

def cmd_init_workdir(args):
    wd = get_workdir(args.env)
    ensure_workdir_layout(wd)
    print(f"[OK] workdir initialized: {wd}")

def cmd_train(args):
    wd = get_workdir(args.env)
    ensure_workdir_layout(wd)

    cfg_path = Path(args.config)
    cfg = _load_yaml(cfg_path)

    task = args.task or cfg.get("task", "inverse")
    run_id = args.run_id or _now_run_id(task)
    run_dir = wd / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # load training params
    train_cfg_yaml = _load_yaml(Path(args.train_config))
    tcfg = TrainConfig(
        seed=int(train_cfg_yaml.get("seed", 42)),
        device=str(train_cfg_yaml.get("device", "auto")),
        epochs=int(train_cfg_yaml.get("epochs", 50)),
        batch_size=int(train_cfg_yaml.get("batch_size", 64)),
        lr=float(train_cfg_yaml.get("lr", 1e-3)),
        weight_decay=float(train_cfg_yaml.get("weight_decay", 1e-4)),
        num_workers=int(train_cfg_yaml.get("num_workers", 4)),
        grad_clip=float(train_cfg_yaml.get("grad_clip", 1.0)),
        loss=str(train_cfg_yaml.get("loss", "smoothl1")),
    )

    # resolve dataset paths under workdir
    train_npz = join_workdir(wd, cfg["data"]["train_npz"])
    val_npz   = join_workdir(wd, cfg["data"]["val_npz"])
    if not train_npz.exists() or not val_npz.exists():
        raise FileNotFoundError(
            f"Processed datasets not found.\ntrain={train_npz}\nval={val_npz}\n"
            "Please place processed npz under NFS_WORKDIR/processed/..."
        )

    tr_pair = load_npz_pair(train_npz)
    va_pair = load_npz_pair(val_npz)

    # fit scalers on train
    x_mean, x_std = fit_channel_zscore(tr_pair.X)
    y_mean, y_std = fit_channel_zscore(tr_pair.Y)
    scaler_path = run_dir / "scaler.npz"
    save_scaler(scaler_path, x_mean, x_std, y_mean, y_std, meta={"task": task, "config": str(cfg_path)})

    # normalize
    Xtr = apply_zscore(tr_pair.X, x_mean, x_std)
    Ytr = apply_zscore(tr_pair.Y, y_mean, y_std)
    Xva = apply_zscore(va_pair.X, x_mean, x_std)
    Yva = apply_zscore(va_pair.Y, y_mean, y_std)

    # dataloaders
    train_ds = FieldNpzDataset(Xtr, Ytr)
    val_ds   = FieldNpzDataset(Xva, Yva)
    train_loader = DataLoader(train_ds, batch_size=tcfg.batch_size, shuffle=True, num_workers=tcfg.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=tcfg.batch_size, shuffle=False, num_workers=tcfg.num_workers, pin_memory=True)

    Cin, H, W = Xtr.shape[1], Xtr.shape[2], Xtr.shape[3]
    Cout = Ytr.shape[1]

    model_cfg = cfg.get("model", {"name":"mlp"})
    model = build_model(model_cfg["name"], cin=Cin, cout=Cout, H=H, W=W, cfg=model_cfg)
    loss_fn = build_loss(tcfg.loss)

    # snapshot resolved config
    resolved = {
        "run_id": run_id,
        "task": task,
        "config_file": str(cfg_path),
        "train_config_file": str(args.train_config),
        "data": {"train_npz": str(train_npz), "val_npz": str(val_npz)},
        "channels": cfg.get("channels", {}),
        "model": model_cfg,
        "train": train_cfg_yaml,
    }
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")

    metrics_csv = train_loop(model, loss_fn, train_loader, val_loader, run_dir, tcfg)

    # plot loss
    plot_loss(metrics_csv, run_dir / "plots" / "loss.png")
    print(f"[OK] run saved: {run_dir}")
    print(f"     best ckpt: {run_dir / 'checkpoints' / 'best.pt'}")
    print(f"     scaler:    {scaler_path}")
    print(f"     loss plot: {run_dir / 'plots' / 'loss.png'}")

def cmd_predict(args):
    wd = get_workdir(args.env)
    run_dir = wd / "runs" / args.run
    if not run_dir.exists():
        raise FileNotFoundError(f"Run not found: {run_dir}")

    resolved = _load_yaml(run_dir / "config_resolved.yaml")
    model_cfg = resolved["model"]
    scaler = run_dir / "scaler.npz"
    ckpt = run_dir / "checkpoints" / "best.pt"

    inp = join_workdir(wd, args.input)
    out = join_workdir(wd, args.out)
    out = predict_npz(inp, scaler, ckpt, model_cfg, out, device=args.device)
    print(f"[OK] predicted: {out}")

def cmd_export_csv(args):
    wd = get_workdir(args.env)
    pred = join_workdir(wd, args.pred)
    out  = join_workdir(wd, args.out)
    cfg  = Path(args.config)
    export_inverse_csv(pred, out, cfg)
    print(f"[OK] exported: {out}")

def main():
    p = argparse.ArgumentParser(prog="nfs-ml", description="NFS-ML Suite CLI")
    p.add_argument("--env", default="NFS_WORKDIR", help="Workdir env name (default: NFS_WORKDIR)")

    sp = p.add_subparsers(dest="cmd", required=True)

    p_init = sp.add_parser("init-workdir", help="Create standard workdir layout under $NFS_WORKDIR")
    p_init.set_defaults(func=cmd_init_workdir)

    p_train = sp.add_parser("train", help="Train forward/inverse model")
    p_train.add_argument("--task", default=None, help="forward|inverse (overrides config)")
    p_train.add_argument("--config", required=True, help="Model config yaml (e.g. configs/model_inverse.yaml)")
    p_train.add_argument("--train-config", default="configs/train.yaml", help="Training config yaml (default: configs/train.yaml)")
    p_train.add_argument("--run-id", default=None, help="Run id; default is timestamp-based")
    p_train.set_defaults(func=cmd_train)

    p_pred = sp.add_parser("predict", help="Predict using a trained run")
    p_pred.add_argument("--run", required=True, help="Run id under workdir/runs/")
    p_pred.add_argument("--input", required=True, help="Input npz path relative to workdir, e.g. processed/inverse/val.npz")
    p_pred.add_argument("--out", required=True, help="Output npz path relative to workdir, e.g. exports/pred_val.npz")
    p_pred.add_argument("--device", default="auto", help="auto|cpu|cuda")
    p_pred.set_defaults(func=cmd_predict)

    p_exp = sp.add_parser("export-csv", help="Export inverse prediction to CST-style CSV (first sample)")
    p_exp.add_argument("--pred", required=True, help="Prediction npz relative to workdir, e.g. exports/pred_val.npz")
    p_exp.add_argument("--out", required=True, help="Output csv relative to workdir, e.g. exports/pred_val.csv")
    p_exp.add_argument("--config", default="configs/data.yaml", help="Data config yaml (grid coordinates)")
    p_exp.set_defaults(func=cmd_export_csv)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
