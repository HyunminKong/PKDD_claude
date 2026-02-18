"""Amortized variational encoder for temporal causal state inference.

Produces q(Z_t | Z_{t-1}, x_{≤t}) as a diagonal Gaussian,
maintaining Markov chain structure consistent with the OU prior.

Architecture:
    1. GRU processes observation history x_{≤t} → hidden state h_t
    2. Combine h_t with Z_{t-1} → output (mu_t, logvar_t) for Z_t
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CausalStateEncoder(nn.Module):
    """Amortized variational encoder for q(Z_t | Z_{t-1}, x_{≤t}).

    Uses a GRU to summarize observation history, then combines with
    previous causal state to produce posterior parameters.

    Args:
        num_vars: N (number of sensors)
        hidden_dim: GRU hidden dimension
        num_gru_layers: number of GRU layers
    """

    def __init__(
        self,
        num_vars: int,
        hidden_dim: int = 128,
        num_gru_layers: int = 1,
    ):
        super().__init__()
        self.num_vars = num_vars
        self.hidden_dim = hidden_dim
        N = num_vars

        # GRU: processes (x_t) sequentially → h_t
        self.gru = nn.GRU(
            input_size=N,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
        )

        # Combine GRU hidden state + flattened Z_{t-1} → posterior params
        combine_input = hidden_dim + N * N
        combine_hidden = hidden_dim

        self.posterior_net = nn.Sequential(
            nn.Linear(combine_input, combine_hidden),
            nn.ReLU(),
            nn.Linear(combine_hidden, combine_hidden),
            nn.ReLU(),
        )

        # Separate heads for mu and logvar
        self.mu_head = nn.Linear(combine_hidden, N * N)
        self.logvar_head = nn.Linear(combine_hidden, N * N)

        # Initialize logvar head to produce small variance initially
        nn.init.constant_(self.logvar_head.bias, -2.0)
        nn.init.zeros_(self.logvar_head.weight)

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        z_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-step posterior update.

        Args:
            x_t: (N,) current observation
            h_prev: (num_layers, hidden_dim) GRU hidden state
            z_prev: (N, N) previous causal state (posterior mean from t-1)

        Returns:
            q_mu: (N, N) posterior mean for Z_t
            q_logvar: (N, N) posterior log-variance for Z_t
            h_new: (num_layers, hidden_dim) updated GRU hidden state
        """
        N = self.num_vars

        # GRU step: x_t → h_new
        # GRU expects (batch=1, seq_len=1, N)
        x_in = x_t.view(1, 1, N)
        _, h_new = self.gru(x_in, h_prev)

        # Combine GRU output with previous Z
        h_out = h_new[-1]  # (1, hidden_dim) — last layer, batch=1
        z_flat = z_prev.view(1, N * N)
        combined = torch.cat([h_out, z_flat], dim=-1)  # (1, hidden+N²)

        # Posterior parameters
        features = self.posterior_net(combined)  # (1, combine_hidden)
        q_mu = self.mu_head(features).view(N, N)
        q_logvar = self.logvar_head(features).view(N, N)

        # Clamp logvar for stability
        q_logvar = q_logvar.clamp(-6.0, 4.0)

        return q_mu, q_logvar, h_new

    def init_hidden(self, device: torch.device) -> torch.Tensor:
        """Initialize GRU hidden state to zeros.

        Returns:
            h0: (num_layers, 1, hidden_dim) — batch dim=1
        """
        return torch.zeros(
            self.gru.num_layers, 1, self.hidden_dim,
            device=device,
        )

    def encode_sequence(
        self,
        X_seq: torch.Tensor,
        z_init: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a full sequence to get posterior parameters at each step.

        Args:
            X_seq: (T, N) observation sequence
            z_init: (N, N) initial Z state (e.g., from OU prior mean)

        Returns:
            q_mus: (T, N, N) posterior means
            q_logvars: (T, N, N) posterior log-variances
        """
        T, N = X_seq.shape
        device = X_seq.device

        q_mus = torch.zeros(T, N, N, device=device)
        q_logvars = torch.zeros(T, N, N, device=device)

        h = self.init_hidden(device)
        z_prev = z_init

        for t in range(T):
            q_mu, q_logvar, h = self.forward(X_seq[t], h, z_prev)
            q_mus[t] = q_mu
            q_logvars[t] = q_logvar
            z_prev = q_mu  # use posterior mean as "previous state"

        return q_mus, q_logvars
