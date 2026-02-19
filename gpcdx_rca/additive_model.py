"""Bypass-blocked causal prediction: mean + variance.

Key bypass-blocking rules:
    1. μ̂ is ONLY from A·basis — no context head, no cross-variable shortcut
    2. σ̂² is from target variable j's OWN lag only — no cross-variable input
    3. Cross-variable interaction ONLY through causal coefficients A_t
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────
#  Data alignment (leakage-free)
# ──────────────────────────────────────────────────────────────────────
def build_hist_and_target(Y: torch.Tensor, lag: int):
    """Build input history and one-step-ahead targets.

    Alignment (no leakage):
        X_hist[s] = [y_t, y_{t-1}, ..., y_{t-lag+1}]   (current included)
        Y_next[s] = y_{t+1}                              (prediction target)

    where t = s + lag - 1.

    Args:
        Y: [T, N] time series
        lag: L

    Returns:
        X_hist: [T-lag, N, lag]
        Y_next: [T-lag, N]
    """
    T, N = Y.shape
    m = T - lag  # number of valid prediction steps

    X_hist = torch.zeros(m, N, lag, device=Y.device, dtype=Y.dtype)
    for k in range(lag):
        # k=0 → y_t, k=1 → y_{t-1}, ..., k=lag-1 → y_{t-lag+1}
        X_hist[:, :, k] = Y[lag - 1 - k : T - 1 - k]

    Y_next = Y[lag:]  # y_{t+1} for each s
    return X_hist, Y_next


# ──────────────────────────────────────────────────────────────────────
#  Mean prediction (causal path ONLY)
# ──────────────────────────────────────────────────────────────────────
class CausalMeanOnly(nn.Module):
    """μ̂_j = Σ_{i,k} A^(k)_{j,i,t} · b^(k)_{i,t} + bias_j

    NO context head. NO cross-variable bypass. Only A and basis.

    Args:
        num_vars: N
    """

    def __init__(self, num_vars: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_vars))

    def forward(
        self,
        A: torch.Tensor,
        B_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Compute prediction mean.

        Args:
            A:      [m, N_out, N_in, L] causal coefficients
            B_feat: [m, N_in, L] basis features

        Returns:
            mu: [m, N_out]
        """
        mu = torch.einsum('mjik,mik->mj', A, B_feat) + self.bias
        return mu


# ──────────────────────────────────────────────────────────────────────
#  Variance prediction (own-lag ONLY — no cross-variable input)
# ──────────────────────────────────────────────────────────────────────
class VarHead(nn.Module):
    """σ̂²_j = softplus(f_j(y_{j,t:t-L+1})) + ε

    Shared MLP + per-variable scale/shift. Input is ONLY variable j's lag.

    Args:
        num_vars: N
        lag: L
        hidden: MLP hidden dim
    """

    def __init__(self, num_vars: int, lag: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(lag, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.scale = nn.Parameter(torch.ones(num_vars))
        self.shift = nn.Parameter(torch.zeros(num_vars))

    def forward(self, X_hist: torch.Tensor) -> torch.Tensor:
        """Predict per-variable variance.

        Args:
            X_hist: [m, N, L]

        Returns:
            sigma2: [m, N]
        """
        m, N, L = X_hist.shape
        raw = self.net(X_hist.reshape(m * N, L)).view(m, N)
        logvar = raw * self.scale + self.shift
        sigma2 = F.softplus(logvar) + 1e-6
        return sigma2
