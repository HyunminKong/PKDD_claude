"""OU (Ornstein-Uhlenbeck) prior for lag-specific causal coefficients.

Prior on A^(k)_{j,i,t} ∈ R (scalar per edge per lag):
    p(A_0) = N(0, Q / (1 - ρ²))             (stationary)
    p(A_t | A_{t-1}) = N(ρ · A_{t-1}, Q)     (AR(1) transition)

Parameters ρ ∈ (0,1) and Q > 0 are scalar (shared across all edges/lags).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class OUPrior(nn.Module):
    """OU prior for lag-specific causal coefficients A_t ∈ R^{N×N×L}.

    Args:
        rho_init: initial AR coefficient
        q_init: initial process noise variance
    """

    def __init__(self, rho_init: float = 0.95, q_init: float = 0.01):
        super().__init__()
        rho_logit = math.log(rho_init / (1.0 - rho_init))
        self.raw_rho = nn.Parameter(torch.tensor(rho_logit))
        q_raw = math.log(math.exp(q_init) - 1.0)
        self.raw_q = nn.Parameter(torch.tensor(q_raw))

    @property
    def rho(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_rho)

    @property
    def Q(self) -> torch.Tensor:
        return F.softplus(self.raw_q)

    def kl(
        self,
        A_mu: torch.Tensor,
        A_logvar: torch.Tensor,
    ) -> torch.Tensor:
        """KL(q(A_{0:T}) || p_OU(A_{0:T})), normalized per element.

        Args:
            A_mu:     [m, N, N, L] posterior means
            A_logvar: [m, N, N, L] posterior log-variances

        Returns:
            scalar KL, normalized by total elements (m * N * N * L)
        """
        m = A_mu.shape[0]
        rho = self.rho
        Q = self.Q

        total_kl = torch.tensor(0.0, device=A_mu.device)

        # t=0: stationary prior N(0, Q/(1-ρ²))
        p_var_0 = Q / (1.0 - rho ** 2 + 1e-8)
        q_var_0 = torch.exp(A_logvar[0])
        kl_0 = 0.5 * (
            torch.log(p_var_0 / (q_var_0 + 1e-8) + 1e-8)
            + q_var_0 / (p_var_0 + 1e-8)
            + A_mu[0] ** 2 / (p_var_0 + 1e-8)
            - 1.0
        )
        total_kl = total_kl + kl_0.sum()

        # t>0: transition prior N(ρ · A_{t-1}, Q)
        for t in range(1, m):
            p_mu = rho * A_mu[t - 1].detach()  # posterior mean, detached
            q_var = torch.exp(A_logvar[t])
            kl_t = 0.5 * (
                torch.log(Q / (q_var + 1e-8) + 1e-8)
                + q_var / (Q + 1e-8)
                + (A_mu[t] - p_mu) ** 2 / (Q + 1e-8)
                - 1.0
            )
            total_kl = total_kl + kl_t.sum()

        return total_kl / A_mu.numel()
