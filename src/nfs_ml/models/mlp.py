from __future__ import annotations
import torch
import torch.nn as nn

class MLPGrid(nn.Module):
    """MLP operating on flattened grid tensors.
    Input:  (B, Cin, H, W)
    Output: (B, Cout, H, W)  (same H/W as target)
    """
    def __init__(self, cin: int, cout: int, H: int, W: int, hidden: int = 2048, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        in_dim = cin * H * W
        out_dim = cout * H * W

        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(inplace=True)]
            if dropout and dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
        self.cout, self.H, self.W = cout, H, W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        y = self.net(x.view(b, -1))
        return y.view(b, self.cout, self.H, self.W)
