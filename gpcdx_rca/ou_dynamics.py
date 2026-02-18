"""OU (Ornstein-Uhlenbeck) dynamics for time-varying causal weights.

Discrete AR(1) formulation:
    Z_{t+1} = mu_Z + rho * (Z_t - mu_Z) + eta_t,   eta_t ~ N(0, Q)

Causal weight transform:
    W_t = tanh(Z_t/τ) · softplus(Z_t)   (signed, temperature-annealed)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class OUDynamics(nn.Module):
    """OU process prior over time-varying causal weight states Z_t.

    Parameters:
        num_vars: N (number of sensors)
        rho_init: initial mean-reversion speed (0 < rho < 1)
        q_init: initial process noise (structural variability)
        mu_z_init: initial long-term mean for Z (negative = sparse start)
        tau_init: initial sigmoid gate temperature (warm = smooth)
        tau_final: final sigmoid gate temperature (cold = sharp)
    """

    def __init__(
        self,
        num_vars: int,
        rho_init: float = 0.95,
        q_init: float = 0.01,
        mu_z_init: float = 0.0,
        tau_init: float = 1.0,
        tau_final: float = 0.2,
    ):
        super().__init__()
        self.num_vars = num_vars
        N = num_vars

        # Long-term mean of Z (learnable per edge)
        self.mu_Z = nn.Parameter(torch.full((N, N), mu_z_init))

        # Edge-wise rho in (0, 1) via sigmoid(raw_rho)
        rho_logit = math.log(rho_init / (1.0 - rho_init))
        self.raw_rho = nn.Parameter(torch.full((N, N), rho_logit))

        # Edge-wise Q > 0 via softplus(raw_q)
        q_raw = math.log(math.exp(q_init) - 1.0)
        self.raw_q = nn.Parameter(torch.full((N, N), q_raw))

        # Temperature annealing for sigmoid gate
        self.tau_init = tau_init
        self.tau_final = tau_final
        self.register_buffer("tau", torch.tensor(tau_init))

    def set_tau(self, progress: float):
        """Update sigmoid gate temperature based on training progress.

        Args:
            progress: float in [0, 1] (0 = start of training, 1 = end)
        """
        tau = self.tau_init + progress * (self.tau_final - self.tau_init)
        self.tau.fill_(tau)

    @property
    def rho(self) -> torch.Tensor:
        """Edge-wise mean-reversion coefficient in (0, 1)."""
        return torch.sigmoid(self.raw_rho)

    @property
    def Q(self) -> torch.Tensor:
        """Edge-wise process noise variance (structural variability) > 0."""
        return F.softplus(self.raw_q)

    @property
    def stationary_variance(self) -> torch.Tensor:
        """Edge-wise stationary variance of OU: Q / (1 - rho^2)."""
        rho = self.rho
        return self.Q / (1.0 - rho ** 2 + 1e-8)

    def prior_t0(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Prior distribution for Z_0: p(Z_0) = N(mu_Z, stationary_var).

        Returns:
            mu: (N, N), var: (N, N)
        """
        return self.mu_Z, self.stationary_variance

    def transition(
        self,
        z_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """OU transition prior: p(Z_t | Z_{t-1}).

        Args:
            z_prev: (N, N) previous state

        Returns:
            mu: (N, N) prior mean for Z_t
            var: (N, N) prior variance for Z_t
        """
        rho = self.rho
        mu = self.mu_Z + rho * (z_prev - self.mu_Z)
        var = self.Q
        return mu, var

    def z_to_W(self, z: torch.Tensor) -> torch.Tensor:
        """Transform raw state Z to signed causal weight W.

        We keep sparse-start behavior for negative Z via softplus(Z), while
        allowing inhibitory edges through a signed gate:
            W = tanh(Z / tau) * softplus(Z)

        Behavior:
            Z << 0: softplus(Z)≈0 -> weak negative (near zero; sparse start)
            Z >> 0: tanh≈1, softplus(Z)≈Z -> strong positive
            Z ≈ 0: W≈0
        """
        tau = self.tau.clamp(min=1e-4)
        signed_gate = torch.tanh(z / tau)
        return signed_gate * F.softplus(z)

    def z_to_positive(self, z: torch.Tensor) -> torch.Tensor:
        """Transform raw state Z to non-negative gated weight.

        This is used for signed decomposition:
            W = W_pos - W_neg,
            W_pos = sigmoid(Z_pos / tau) * softplus(Z_pos),
            W_neg = sigmoid(Z_neg / tau) * softplus(Z_neg).
        """
        tau = self.tau.clamp(min=1e-4)
        gate = torch.sigmoid(z / tau)
        return gate * F.softplus(z)

    def kl_t0(
        self,
        q_mu: torch.Tensor,
        q_logvar: torch.Tensor,
    ) -> torch.Tensor:
        """KL(q(Z_0) || p(Z_0)) where both are diagonal Gaussians.

        Args:
            q_mu: (N, N) posterior mean
            q_logvar: (N, N) posterior log-variance
        """
        p_mu, p_var = self.prior_t0()
        return self._kl_diagonal(q_mu, q_logvar, p_mu, p_var)

    def kl_transition(
        self,
        q_mu: torch.Tensor,
        q_logvar: torch.Tensor,
        z_prev: torch.Tensor,
    ) -> torch.Tensor:
        """KL(q(Z_t) || p(Z_t | Z_{t-1})).

        Args:
            q_mu: (N, N) posterior mean at t
            q_logvar: (N, N) posterior log-variance at t
            z_prev: (N, N) previous state (posterior mean from t-1)
        """
        p_mu, p_var = self.transition(z_prev)
        return self._kl_diagonal(q_mu, q_logvar, p_mu, p_var)

    @staticmethod
    def _kl_diagonal(
        q_mu: torch.Tensor,
        q_logvar: torch.Tensor,
        p_mu: torch.Tensor,
        p_var: torch.Tensor,
    ) -> torch.Tensor:
        """KL(N(q_mu, q_var) || N(p_mu, p_var)), summed over all elements."""
        q_var = torch.exp(q_logvar)
        # KL = 0.5 * [log(p_var/q_var) + q_var/p_var + (q_mu-p_mu)^2/p_var - 1]
        kl = 0.5 * (
            torch.log(p_var / (q_var + 1e-8) + 1e-8)
            + q_var / (p_var + 1e-8)
            + (q_mu - p_mu) ** 2 / (p_var + 1e-8)
            - 1.0
        )
        return kl.sum()
