from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import normalize_adjacency


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, X, A_norm):
        # X: (..., N, F), A_norm: (N, N)  — broadcasts over batch dims
        return self.lin(A_norm @ X)


class VGAE(nn.Module):
    """Variational Graph AutoEncoder for exogenous latent variables (Section 4.3).

    Encoder (Eq 18):
      H_1       = ReLU(Ã X_enc W_1)        — 1 shared GCN layer
      mu_u      = Ã H_1 W_mu               — GCN head (no activation)
      log σ²_u  = Ã H_1 W_{log σ}          — GCN head (no activation)

    Decoder (Eq 22):
      H_2  = ReLU(Ã Z W_2)                 — GCN + ReLU
      Ĥ    = Ã H_2 W_out                   — GCN output (no activation)
    """

    def __init__(self, P: int, N: int, d: int, hidden: int):
        super().__init__()
        self.P = P
        self.N = N
        self.d = d

        # Encoder (Eq 18): 1 shared GCN + 2 GCN heads
        enc_in = P + N + 1
        self.enc_gcn1 = GCNLayer(enc_in, hidden)       # shared
        self.enc_mu = GCNLayer(hidden, d)               # GCN head for mu
        self.enc_logvar = GCNLayer(hidden, d)           # GCN head for logvar

        # Decoder (Eq 22): 2 GCN layers
        dec_in = P + d + 2
        self.dec_gcn1 = GCNLayer(dec_in, hidden)        # GCN + ReLU
        self.dec_out = GCNLayer(hidden, 1)              # GCN output (no ReLU)

    def encode(self, X_enc, A_norm):
        """
        X_enc:  (B, N, P+N+1)  or  (N, P+N+1)
        A_norm: (N, N)
        Returns mu_u, logvar_u: same leading dims + (d,)
        """
        h = F.relu(self.enc_gcn1(X_enc, A_norm))        # H_1
        return self.enc_mu(h, A_norm), self.enc_logvar(h, A_norm)

    def decode(self, Z, A_norm):
        """
        Z:      (..., N, P+d+2)
        Returns x_hat: (..., N)
        """
        h = F.relu(self.dec_gcn1(Z, A_norm))            # H_2
        return self.dec_out(h, A_norm).squeeze(-1)       # Ĥ (no ReLU)

    def forward(self, X_enc, A_norm, h_t, mu_sum, log_sigma_sum):
        """
        Args
        ----
        X_enc:          (B, N, P+N+1)  encoder input (Eq 18)
        A_norm:         (N, N)          fixed normalized adjacency
        h_t:            (B, N, P)       local history per node
        mu_sum:         (B, N)          Eq 17 summary mean
        log_sigma_sum:  (B, N)          Eq 17 summary log-variance

        Returns
        -------
        x_hat:    (B, N)      reconstructed x_t
        mu_u:     (B, N, d)   exogenous posterior mean
        logvar_u: (B, N, d)   exogenous posterior log-variance
        """
        mu_u, logvar_u = self.encode(X_enc, A_norm)

        # Reparameterization (Eq 20)
        std = torch.exp(0.5 * logvar_u)
        u_t = mu_u + std * torch.randn_like(std)

        # Decoder input (Eq 21): [h_t; u_t; mu_sum; log_sigma_sum]
        Z = torch.cat([h_t, u_t,
                       mu_sum.unsqueeze(-1),
                       log_sigma_sum.unsqueeze(-1)], dim=-1)

        x_hat = self.decode(Z, A_norm)
        return x_hat, mu_u, logvar_u
