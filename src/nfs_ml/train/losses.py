from __future__ import annotations
import torch.nn as nn

def build_loss(name: str) -> nn.Module:
    name = name.lower().strip()
    if name == "mse":
        return nn.MSELoss()
    if name == "smoothl1":
        return nn.SmoothL1Loss()
    raise ValueError(f"Unknown loss: {name}")
