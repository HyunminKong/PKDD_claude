"""Univariate lag basis transformation (fixed structure).

Always: b(y) = y + MLP(y)  (identity skip + small nonlinear residual)

This guarantees:
    - Linear (VAR) solution is always reachable (via identity path)
    - Nonlinear data can be captured (via MLP residual)
    - No dataset-specific mode switching needed
    - Cross-variable interaction is ONLY through causal coefficients A
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LagBasis(nn.Module):
    """Per-element univariate basis: scalar -> scalar, with identity skip.

    b(y) = y + MLP(y)

    The identity path ensures VAR-like solutions are always available.
    The MLP residual adds nonlinear capacity (kept small to avoid bypass).

    Args:
        hidden: MLP hidden dim (kept small to limit capacity)
    """

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # Initialize MLP output near zero so basis starts as identity
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, X_hist: torch.Tensor) -> torch.Tensor:
        """Transform lag values: b(y) = y + MLP(y).

        Args:
            X_hist: [m, N, L] raw lag values

        Returns:
            B_feat: [m, N, L] transformed basis features
        """
        shape = X_hist.shape
        mlp_out = self.mlp(X_hist.reshape(-1, 1)).reshape(shape)
        return X_hist + mlp_out
