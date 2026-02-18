"""Low-rank + diagonal residual covariance model.

Residual:  r_t = x_t - f_θ(·; W_t)

Covariance structure:
    Σ_r = B B⊤ + D

where:
    B: (N, k)  — low-rank factor (common exogenous modes)
    D: diag(σ²_{e,1}, ..., σ²_{e,N})  — node-specific noise

This constrains the residual to have at most k correlated modes,
preventing the exogenous from explaining everything.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankResidualModel(nn.Module):
    """Low-rank + diagonal Gaussian model for residuals.

    Args:
        num_vars: N (number of sensors)
        rank: k (number of common factors, k << N)
    """

    def __init__(self, num_vars: int, rank: int = 2):
        super().__init__()
        self.num_vars = num_vars
        self.rank = rank

        # Low-rank factor B: (N, k)
        self.B = nn.Parameter(torch.randn(num_vars, rank) * 0.1)

        # Diagonal noise D: softplus(raw_d) for positivity
        # Initialize larger (softplus(2)≈2.13) to absorb large initial residuals
        self.raw_d = nn.Parameter(torch.full((num_vars,), 2.0))

    @property
    def D_diag(self) -> torch.Tensor:
        """Diagonal noise variances (N,)."""
        return F.softplus(self.raw_d) + 1e-4

    @property
    def covariance(self) -> torch.Tensor:
        """Full covariance Σ = B B⊤ + D.  (N, N)"""
        return self.B @ self.B.t() + torch.diag(self.D_diag)

    def log_prob(self, r: torch.Tensor) -> torch.Tensor:
        """Log-probability of residuals under N(0, BB⊤ + D).

        Uses Woodbury identity for efficient computation:
            (BB⊤ + D)^{-1} = D^{-1} - D^{-1}B(I + B⊤D^{-1}B)^{-1}B⊤D^{-1}
            log|BB⊤ + D| = log|I + B⊤D^{-1}B| + Σ log D_ii

        Args:
            r: (batch, N) or (N,) residual vectors

        Returns:
            total_log_prob: scalar (summed over batch)
        """
        if r.dim() == 1:
            r = r.unsqueeze(0)

        B, N = r.shape
        k = self.rank

        D_inv = (1.0 / self.D_diag).clamp(max=1e4)  # (N,)
        D_inv_r = r * D_inv.unsqueeze(0)        # (B, N)

        # M = I_k + B⊤ D^{-1} B   — (k, k)
        BtDinv = self.B.t() * D_inv.unsqueeze(0)  # (k, N)
        M = torch.eye(k, device=r.device) + BtDinv @ self.B  # (k, k)

        # NaN/Inf guard: fall back to diagonal-only if M is corrupted
        if not torch.isfinite(M).all():
            quad = (r ** 2 * D_inv.unsqueeze(0)).sum(dim=-1)
            log_det = torch.log(self.D_diag).sum()
            log_prob = -0.5 * (N * math.log(2 * math.pi) + log_det + quad)
            return log_prob.sum()

        # Cholesky with progressive jitter
        L_M = None
        for jitter in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
            try:
                M_j = M + jitter * torch.eye(k, device=r.device)
                L_M = torch.linalg.cholesky(M_j)
                break
            except torch._C._LinAlgError:
                continue

        if L_M is None:
            # All jitters failed — fall back to diagonal-only
            quad = (r ** 2 * D_inv.unsqueeze(0)).sum(dim=-1)
            log_det = torch.log(self.D_diag).sum()
            log_prob = -0.5 * (N * math.log(2 * math.pi) + log_det + quad)
            return log_prob.sum()

        # Woodbury: Σ^{-1} r = D^{-1}r - D^{-1}B M^{-1} B⊤ D^{-1}r
        BtDinv_r = (BtDinv @ r.t())         # (k, B)
        M_inv_BtDinv_r = torch.cholesky_solve(
            BtDinv_r, L_M)                   # (k, B)
        correction = (self.B @ M_inv_BtDinv_r).t()  # (B, N)
        Sigma_inv_r = D_inv_r - correction * D_inv.unsqueeze(0)

        # Quadratic form: r⊤ Σ^{-1} r
        quad = (r * Sigma_inv_r).sum(dim=-1)  # (B,)

        # Log-determinant: log|Σ| = log|M| + Σ log D_ii
        log_det_M = 2.0 * torch.log(torch.diag(L_M)).sum()
        log_det_D = torch.log(self.D_diag).sum()
        log_det = log_det_M + log_det_D

        # log N(r; 0, Σ) = -0.5 * (N log(2π) + log|Σ| + r⊤Σ^{-1}r)
        log_prob = -0.5 * (N * math.log(2 * math.pi) + log_det + quad)

        return log_prob.sum()

    def nll_per_sample(self, r: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood per sample under N(0, BB⊤ + D).

        Args:
            r: (batch, N) or (N,) residual vectors

        Returns:
            nll: (batch,) per-sample NLL values
        """
        if r.dim() == 1:
            r = r.unsqueeze(0)

        batch, N = r.shape
        k = self.rank

        D_inv = (1.0 / self.D_diag).clamp(max=1e4)  # (N,)
        D_inv_r = r * D_inv.unsqueeze(0)             # (B, N)

        BtDinv = self.B.t() * D_inv.unsqueeze(0)     # (k, N)
        M = torch.eye(k, device=r.device) + BtDinv @ self.B  # (k, k)

        # Fallback to diagonal model if numerical issues arise
        if not torch.isfinite(M).all():
            quad_diag = (r ** 2 * D_inv.unsqueeze(0)).sum(dim=-1)
            log_det_diag = torch.log(self.D_diag).sum()
            return 0.5 * (N * math.log(2 * math.pi) + log_det_diag + quad_diag)

        L_M = None
        for jitter in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
            try:
                M_j = M + jitter * torch.eye(k, device=r.device)
                L_M = torch.linalg.cholesky(M_j)
                break
            except torch._C._LinAlgError:
                continue

        if L_M is None:
            quad_diag = (r ** 2 * D_inv.unsqueeze(0)).sum(dim=-1)
            log_det_diag = torch.log(self.D_diag).sum()
            return 0.5 * (N * math.log(2 * math.pi) + log_det_diag + quad_diag)

        BtDinv_r = (BtDinv @ r.t())  # (k, B)
        M_inv_BtDinv_r = torch.cholesky_solve(BtDinv_r, L_M)  # (k, B)
        correction = (self.B @ M_inv_BtDinv_r).t()  # (B, N)
        Sigma_inv_r = D_inv_r - correction * D_inv.unsqueeze(0)
        quad = (r * Sigma_inv_r).sum(dim=-1)  # (B,)

        log_det_M = 2.0 * torch.log(torch.diag(L_M)).sum()
        log_det_D = torch.log(self.D_diag).sum()
        log_det = log_det_M + log_det_D

        return 0.5 * (N * math.log(2 * math.pi) + log_det + quad)

    def mahalanobis_per_node(self, r: torch.Tensor) -> torch.Tensor:
        """Per-node Mahalanobis-like scores for RCA.

        Decomposes residual into common factor + individual:
            s_t = (B⊤B + I)^{-1} B⊤ r_t   (MAP of common factor)
            e_t = r_t - B s_t               (individual residual)
            score_i = e_{i,t}² / D_ii       (node-level anomaly)

        Args:
            r: (batch, N) or (N,) residuals

        Returns:
            node_scores: (batch, N) per-node anomaly scores
        """
        if r.dim() == 1:
            r = r.unsqueeze(0)

        # MAP estimate of common factor s
        BtB = self.B.t() @ self.B  # (k, k)
        M = BtB + torch.eye(self.rank, device=r.device)
        Bt_r = (r @ self.B)  # (batch, k)
        s = torch.linalg.solve(M, Bt_r.t()).t()  # (batch, k)

        # Individual residual
        e = r - s @ self.B.t()  # (batch, N)

        # Per-node normalized score
        node_scores = e ** 2 / (self.D_diag.unsqueeze(0) + 1e-8)
        return node_scores
