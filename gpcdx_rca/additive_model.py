"""Neural Granger prediction model with vector features and additive aggregation.

Prediction for target i at time t:
    φ_j = SharedEncoder(x_{j, t-1:t-p}) ∈ R^d  (source feature vector)
    agg_i = Σ_j W_{ij,t} · φ_j                  (W-gated sum, ALL j including j=i)
    x̂_{i,t} = v_i · agg_i                       (per-target linear readout)

Key design choices:
    - SUM aggregation (not MLP decoder) ensures W is identifiable:
      the model MUST assign correct W to achieve low prediction loss.
    - Self-dynamics (j=i) are handled by W_{ii} — no separate AR path.
      This forces W to learn the full causal structure including self-loops.
    - Vector features (d > 1) provide richer representations than
      scalar-output source modules, improving nonlinear fitting.
    - Per-target readout v_i allows each target to attend to different
      aspects of source features without breaking W identifiability.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class NeuralGrangerModel(nn.Module):
    """Neural Granger prediction: vector features + additive W-gated aggregation.

    Architecture:
        - SharedEncoder: one MLP mapping R^lag → R^d (shared across all sources)
        - Additive aggregation: agg_i = Σ_j W_{ij} · φ_j  (W is the bottleneck)
        - Per-target readout: c_i = v_i · agg_i (linear projection, no decoder MLP)
        - NO separate AR modules — self-dynamics go through W_{ii}

    Args:
        num_vars: N (number of sensors)
        lag: p (lag window size)
        feat_dim: d (source feature dimension)
        hidden_dim: hidden layer width for encoder
    """

    def __init__(
        self,
        num_vars: int,
        lag: int,
        feat_dim: int = 8,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.num_vars = num_vars
        self.lag = lag
        self.feat_dim = feat_dim

        # Shared source encoder: R^lag → R^d
        self.source_encoder = nn.Sequential(
            nn.Linear(lag, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim),
        )

        # EMA-based source-wise standardization.
        self.register_buffer('feat_running_mean', torch.zeros(num_vars, feat_dim))
        self.register_buffer('feat_running_var', torch.ones(num_vars, feat_dim))
        self.feat_momentum = 0.1

        # Per-target readout: each target selects which feature dimensions
        # are relevant via v_i ∈ R^d (linear projection, no bias)
        self.readout_v = nn.Parameter(torch.randn(num_vars, feat_dim) * 0.1)

    def forward(
        self,
        lag_windows: torch.Tensor,
        W: torch.Tensor,
    ) -> torch.Tensor:
        """Predict x̂_i = v_i · (Σ_j W_{ij} · φ_j).

        All prediction goes through W, including self-dynamics (j=i).

        Args:
            lag_windows: (batch, N, lag) — lagged inputs for each source
            W: (N, N) or (batch, N, N) — full causal weight matrix

        Returns:
            x_hat: (batch, N) — predicted values for all targets
        """
        B, N, P = lag_windows.shape

        # --- Source features: shared encoder across all sources ---
        phi_raw = self.source_encoder(
            lag_windows.reshape(B * N, P)
        ).reshape(B, N, self.feat_dim)

        # EMA source-wise standardization
        if self.training:
            with torch.no_grad():
                batch_mean = phi_raw.mean(dim=0)
                batch_var = phi_raw.var(dim=0, unbiased=False)
                self.feat_running_mean.mul_(1 - self.feat_momentum).add_(self.feat_momentum * batch_mean)
                self.feat_running_var.mul_(1 - self.feat_momentum).add_(self.feat_momentum * batch_var)

        phi = (
            (phi_raw - self.feat_running_mean.unsqueeze(0))
            / (self.feat_running_var.unsqueeze(0).sqrt() + 1e-6)
        )

        # --- Causal aggregation (full matrix, including diagonal) ---
        if W.dim() == 2:
            W_3d = W.unsqueeze(0).expand(B, -1, -1)
        else:
            W_3d = W

        # (B, N_tgt, N_src) @ (B, N_src, d) → (B, N_tgt, d)
        agg = torch.bmm(W_3d, phi)

        # Per-target linear readout: (B, N, d) * (1, N, d) → sum over d → (B, N)
        x_hat = (agg * self.readout_v.unsqueeze(0)).sum(dim=-1)

        return x_hat

    def build_lag_windows(
        self,
        X_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build lag windows and targets from a raw time series.

        Args:
            X_seq: (T, N) raw time series

        Returns:
            lag_windows: (T-lag, N, lag)
            targets: (T-lag, N)
        """
        T, N = X_seq.shape
        lag = self.lag

        windows = []
        for t in range(lag, T):
            w = X_seq[t - lag:t].flip(0).t()  # (N, lag)
            windows.append(w)

        lag_windows = torch.stack(windows, dim=0)  # (T-lag, N, lag)
        targets = X_seq[lag:]                       # (T-lag, N)
        return lag_windows, targets
