from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
import torch.nn as nn
from .mlp import MLPGrid

def build_model(name: str, *, cin: int, cout: int, H: int, W: int, cfg: Dict[str, Any]) -> nn.Module:
    name = name.lower().strip()
    if name == "mlp":
        return MLPGrid(
            cin=cin, cout=cout, H=H, W=W,
            hidden=int(cfg.get("hidden", 2048)),
            depth=int(cfg.get("depth", 4)),
            dropout=float(cfg.get("dropout", 0.1)),
        )
    raise ValueError(f"Unknown model name: {name}")
