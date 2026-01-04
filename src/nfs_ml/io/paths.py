from __future__ import annotations
import os
from pathlib import Path

def get_workdir(env_name: str = "NFS_WORKDIR") -> Path:
    v = os.environ.get(env_name, "").strip()
    if not v:
        raise RuntimeError(
            f"Environment variable {env_name} is not set. "
            f"Please set it to a directory outside the repo."
        )
    return Path(v).expanduser().resolve()

def ensure_workdir_layout(workdir: Path) -> None:
    # Do NOT create under repo. Workdir should be separate.
    (workdir / "raw").mkdir(parents=True, exist_ok=True)
    (workdir / "processed").mkdir(parents=True, exist_ok=True)
    (workdir / "splits").mkdir(parents=True, exist_ok=True)
    (workdir / "runs").mkdir(parents=True, exist_ok=True)
    (workdir / "exports").mkdir(parents=True, exist_ok=True)
    (workdir / "cache").mkdir(parents=True, exist_ok=True)

def join_workdir(workdir: Path, rel: str) -> Path:
    # A safe join: user passes "processed/inverse/train.npz" etc.
    rel = rel.replace("\\", "/").lstrip("/")
    return (workdir / rel).resolve()
