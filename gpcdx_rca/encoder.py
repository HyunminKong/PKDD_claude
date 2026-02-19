"""Factorized inference network for posterior q(A_t).

Produces q(A_t) = N(mu_{A,t}, diag(sigma^2_{A,t})) where A_t in R^{N x N x L}.

Architecture:
    1. GRU processes flattened X_hist = [y_t, ..., y_{t-L+1}] -> h_t
    2. Information bottleneck: h_t -> z_t (small dim) to limit per-step fitting
    3. Learned embeddings for (target j, source i, lag k)
    4. Factorized MLP: z_t + e_j + e_i + e_k -> (mu, logvar) per edge

The bottleneck prevents the InferenceNet from using A_t as a free
per-timestep fitting parameter, forcing reliance on OU prior for
temporal structure.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InferenceNet(nn.Module):
    """Factorized posterior q(A_t) with temporal bottleneck.

    Args:
        num_vars: N
        lag: L
        gru_hidden: GRU hidden dimension H
        bottleneck_dim: information bottleneck dim (limits per-step fitting)
        emb_dim: embedding dimension E for (j, i, k)
        mlp_hidden: shared MLP hidden dimension
    """

    def __init__(
        self,
        num_vars: int,
        lag: int,
        gru_hidden: int = 128,
        bottleneck_dim: int = 8,
        emb_dim: int = 16,
        mlp_hidden: int = 64,
    ):
        super().__init__()
        self.num_vars = num_vars
        self.lag = lag
        self.mlp_hidden = mlp_hidden

        # GRU: temporal context from observation history
        self.gru = nn.GRU(num_vars * lag, gru_hidden, batch_first=True)

        # Information bottleneck: compress h_t to limit per-step capacity
        self.h_bottleneck = nn.Sequential(
            nn.Linear(gru_hidden, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Learned embeddings for target, source, lag
        self.e_out = nn.Embedding(num_vars, emb_dim)  # target j
        self.e_in = nn.Embedding(num_vars, emb_dim)   # source i
        self.e_lag = nn.Embedding(lag, emb_dim)        # lag k

        # Factorized projection: bottleneck + edge_emb -> hidden (additive)
        self.h_proj = nn.Linear(bottleneck_dim, mlp_hidden)
        self.e_proj = nn.Linear(3 * emb_dim, mlp_hidden)

        # Output: hidden -> (mu, logvar)
        self.out_proj = nn.Linear(mlp_hidden, 2)

        # Initialize logvar output for small initial variance
        nn.init.zeros_(self.out_proj.weight[1])
        nn.init.constant_(self.out_proj.bias[1], -2.0)

    def _build_edge_embeddings(self, device: torch.device) -> torch.Tensor:
        """Build combined edge embeddings for all (j, i, k) triples.

        Returns:
            edge_emb: [N*N*L, 3E] concatenated embeddings
        """
        N, L = self.num_vars, self.lag
        j_idx = torch.arange(N, device=device)
        i_idx = torch.arange(N, device=device)
        k_idx = torch.arange(L, device=device)

        jj, ii, kk = torch.meshgrid(j_idx, i_idx, k_idx, indexing='ij')
        edge_emb = torch.cat([
            self.e_out(jj.reshape(-1)),
            self.e_in(ii.reshape(-1)),
            self.e_lag(kk.reshape(-1)),
        ], dim=-1)  # [N*N*L, 3E]
        return edge_emb

    def forward(self, X_hist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Infer posterior q(A_t) for each timestep.

        Args:
            X_hist: [m, N, L] observation history (y_t ... y_{t-L+1})

        Returns:
            A_mu:     [m, N, N, L] posterior mean
            A_logvar: [m, N, N, L] posterior log-variance
        """
        m, N, L = X_hist.shape
        device = X_hist.device

        # GRU over time
        gru_in = X_hist.reshape(m, N * L).unsqueeze(0)  # [1, m, N*L]
        h_seq, _ = self.gru(gru_in)   # [1, m, H]
        h = h_seq.squeeze(0)          # [m, H]

        # Information bottleneck (limits per-step fitting capacity)
        h_bn = self.h_bottleneck(h)  # [m, bottleneck_dim]

        # Edge embeddings (computed once, reused)
        edge_emb = self._build_edge_embeddings(device)  # [n_edges, 3E]

        # Factorized projection (additive -- avoids large concat tensor)
        h_feat = self.h_proj(h_bn)      # [m, mlp_hidden]
        e_feat = self.e_proj(edge_emb)  # [n_edges, mlp_hidden]

        # Broadcast add: [m, 1, D] + [1, n_edges, D] -> [m, n_edges, D]
        combined = F.relu(h_feat.unsqueeze(1) + e_feat.unsqueeze(0))

        # Output
        out = self.out_proj(combined.reshape(-1, self.mlp_hidden))
        out = out.view(m, N, N, L, 2)

        A_mu = out[..., 0]
        A_logvar = out[..., 1].clamp(-6.0, 4.0)

        return A_mu, A_logvar
