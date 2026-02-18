from __future__ import annotations

import math
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
#  Deep Feature Extractor (Deep Kernel Learning)
# ======================================================================

class DeepFeatureExtractor(nn.Module):
    """MLP feature extractor for Deep Kernel Learning.

    Maps raw lagged time-series window R^P -> R^d_feat so that the RBF
    kernel operates in a learned low-dimensional space instead of the
    raw (potentially very high-dimensional) lag space.

    Shared across all N target GPs for parameter efficiency.
    """

    def __init__(self, input_dim: int, feat_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, feat_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., P) -> (..., d_feat)"""
        return self.net(x)


# ======================================================================
#  RBF Kernel
# ======================================================================

class RBFKernel(nn.Module):
    """RBF kernel with fixed variance: k(x,x') = exp(-1/(2 ell^2) ||x-x'||^2)

    Variance is fixed to 1.0 so that the causal weight W[i,j]^2 is the
    sole scale parameter per source, ensuring identifiability.
    """

    def __init__(self, lengthscale: float = 1.0):
        super().__init__()
        self.log_lengthscale = nn.Parameter(
            torch.tensor(math.log(lengthscale), dtype=torch.float))

    def forward(self, X: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        ls2 = torch.exp(self.log_lengthscale) ** 2
        # X: (n, d), Z: (m, d)
        X2 = (X ** 2).sum(-1, keepdim=True)
        Z2 = (Z ** 2).sum(-1, keepdim=True).transpose(0, 1)
        cross = X @ Z.transpose(0, 1)
        dist2 = X2 + Z2 - 2.0 * cross
        return torch.exp(-0.5 * dist2 / ls2.clamp(min=1e-6))


# ======================================================================
#  Per-target GP with Multiple Kernel Learning (MKL)
# ======================================================================

class PerTargetMKL(nn.Module):
    """GP for one target variable using Multiple Kernel Learning.

    The kernel is an additive combination of per-source RBF kernels::

        K_combined(X, X') = Σ_j  W[j] · K_j(Z_j, Z_j')

    where K_j is an independent RBF kernel operating on source j's
    deep features in R^d_feat.

    Key advantages over single-kernel gated concatenation:
      * Each kernel K_j operates in d_feat space (not N·d_feat),
        avoiding the curse of dimensionality.
      * W directly scales per-source kernel contributions — when
        W[j]=0, source j's kernel is completely removed.
      * Each source has an independent lengthscale hyperparameter,
        allowing the GP to adapt smoothness per source.  Kernel variance
        is fixed to 1.0 — W^2 is the sole scale parameter (identifiable).
    """

    def __init__(self, num_sources: int):
        super().__init__()
        self.num_sources = num_sources
        self.kernels = nn.ModuleList([RBFKernel() for _ in range(num_sources)])
        self.log_noise = nn.Parameter(
            torch.tensor(math.log(1e-2), dtype=torch.float))
        # Cached posterior (populated by cache_posterior)
        self.Z_trains: List[torch.Tensor] | None = None  # list of (n, d_feat)
        self.alpha: torch.Tensor | None = None             # (n, 1)
        self.L: torch.Tensor | None = None                 # (n, n)

    # ---- Robust Cholesky with adaptive jitter ----
    @staticmethod
    def _robust_cholesky(M: torch.Tensor, max_tries: int = 6) -> torch.Tensor:
        jitter = 1e-6
        for _ in range(max_tries):
            try:
                return torch.linalg.cholesky(
                    M + jitter * torch.eye(M.shape[0], dtype=M.dtype, device=M.device))
            except RuntimeError:
                jitter *= 10.0
        return torch.linalg.cholesky(
            M + 1e-1 * torch.eye(M.shape[0], dtype=M.dtype, device=M.device))

    # ---- Additive kernel with effective weight scaling ----
    def combined_kernel(
        self,
        Z_list_a: List[torch.Tensor],
        Z_list_b: List[torch.Tensor],
        W_row: torch.Tensor,
    ) -> torch.Tensor:
        """K = Σ_j W_eff[j] · K_j(Z_a_j, Z_b_j)

        W_eff[j] = z[j] * a[j]  (edge gate × edge strength) from
        the Bayesian causal weight sampling.

        Args:
            Z_list_a: list of (n_a, d_feat) per source
            Z_list_b: list of (n_b, d_feat) per source
            W_row:    (N_src,) effective causal weights for this target
        """
        K = torch.zeros(
            Z_list_a[0].shape[0], Z_list_b[0].shape[0],
            dtype=Z_list_a[0].dtype, device=Z_list_a[0].device,
        )
        for j in range(self.num_sources):
            K = K + W_row[j] * self.kernels[j](Z_list_a[j], Z_list_b[j])
        return K

    # ---- MLL ----
    def mll(
        self,
        Z_list: List[torch.Tensor],
        y: torch.Tensor,
        W_row: torch.Tensor,
    ) -> torch.Tensor:
        K = self.combined_kernel(Z_list, Z_list, W_row)
        n = y.shape[0]
        noise = torch.exp(self.log_noise)
        L = self._robust_cholesky(K + noise * torch.eye(n, dtype=K.dtype, device=K.device))
        alpha = torch.cholesky_solve(y.view(-1, 1), L)
        log_det = 2.0 * torch.log(torch.diag(L)).sum()
        return (-0.5 * y.view(1, -1) @ alpha
                - 0.5 * log_det
                - 0.5 * n * math.log(2 * math.pi))

    # ---- Cache posterior ----
    def cache_posterior(
        self,
        Z_list: List[torch.Tensor],
        y: torch.Tensor,
        W_row: torch.Tensor,
    ):
        with torch.no_grad():
            K = self.combined_kernel(Z_list, Z_list, W_row)
            n = K.shape[0]
            noise = torch.exp(self.log_noise)
            L = self._robust_cholesky(K + noise * torch.eye(n, dtype=K.dtype, device=K.device))
            self.Z_trains = [Z.detach().clone() for Z in Z_list]
            self.L = L.detach()
            self.alpha = torch.cholesky_solve(y.view(-1, 1), L).detach()

    # ---- Prediction (differentiable through z_star features) ----
    def predict(
        self,
        Z_list_star: List[torch.Tensor],
        W_row: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict mean and variance.

        Gradients flow through *Z_list_star* and *W_row*.
        Cached Z_trains / alpha / L are treated as constants.

        Args:
            Z_list_star: list of (m, d_feat) per source
            W_row: (N_src,) causal weights
        Returns:
            mean: (m,), var: (m,)
        """
        Kx = self.combined_kernel(Z_list_star, self.Z_trains, W_row)  # (m, n)
        mean = (Kx @ self.alpha).view(-1)
        v = torch.linalg.solve_triangular(
            self.L, Kx.transpose(0, 1), upper=False)                  # (n, m)
        Kss = self.combined_kernel(
            Z_list_star, Z_list_star, W_row).diagonal()
        var = Kss - (v ** 2).sum(dim=0) + torch.exp(self.log_noise)
        return mean, var.clamp(min=1e-9)

    # ---- Fully differentiable prediction (recomputes posterior) ----
    def predict_differentiable(
        self,
        Z_list_train: List[torch.Tensor],
        y_train: torch.Tensor,
        Z_list_star: List[torch.Tensor],
        W_row: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fully differentiable prediction — recomputes posterior from scratch.

        Unlike predict(), this does NOT use cached (alpha, L, Z_trains).
        Gradients flow through all inputs including training features
        and causal weights used in the training kernel matrix.

        Also returns MLL (at no extra cost since K, L, alpha are already
        computed), allowing Phase 2 to include the causal learning signal.

        Args:
            Z_list_train: list of (n, d_feat) training features per source
            y_train: (n,) training targets
            Z_list_star: list of (m, d_feat) test features per source
            W_row: (N_src,) causal weights for this target
        Returns:
            mean: (m,), var: (m,), mll: scalar
        """
        K = self.combined_kernel(Z_list_train, Z_list_train, W_row)
        n = y_train.shape[0]
        noise = torch.exp(self.log_noise)
        L = self._robust_cholesky(
            K + noise * torch.eye(n, dtype=K.dtype, device=K.device))
        alpha = torch.cholesky_solve(y_train.view(-1, 1), L)

        # MLL (reuses K, L, alpha — zero extra cost)
        log_det = 2.0 * torch.log(torch.diag(L)).sum()
        mll = (-0.5 * y_train.view(1, -1) @ alpha
               - 0.5 * log_det
               - 0.5 * n * math.log(2 * math.pi))

        Kx = self.combined_kernel(Z_list_star, Z_list_train, W_row)  # (m, n)
        mean = (Kx @ alpha).view(-1)
        v = torch.linalg.solve_triangular(
            L, Kx.transpose(0, 1), upper=False)                       # (n, m)
        Kss = self.combined_kernel(
            Z_list_star, Z_list_star, W_row).diagonal()
        var = Kss - (v ** 2).sum(dim=0) + noise
        return mean, var.clamp(min=1e-9), mll.squeeze()


# ======================================================================
#  Granger-GP-VAR with Bayesian Causal Weights (Edge Gate + Strength)
# ======================================================================

class GrangerGPVAR(nn.Module):
    """Bayesian Granger causality with edge gate z + edge strength a.

    Architecture: Multiple Kernel Learning (MKL) + Bayesian Causal Weights
    ──────────────────────────────────────────────────────────────────────
    For each target variable *i*, a single GP predicts x_i(t) using
    an additive kernel over per-source RBF kernels::

        K_i(x, x') = Σ_j  z[i,j] · a[i,j] · K̃_j( g(x_j), g(x_j') )

    where:
      - z[i,j] ∈ {0,1}: edge existence (Concrete/Gumbel-sigmoid relaxation)
      - a[i,j] ≥ 0:     edge strength  (Softplus-Normal variational posterior)
      - K̃_j:            RBF kernel with fixed variance (shape only)
      - g:               shared deep feature extractor

    Training via ELBO:
      E_q[log p(y | K_i)] - KL(q(z,a) || p(z,a))
    """

    def __init__(
        self,
        num_vars: int,
        lag: int,
        feat_dim: int = 16,
        feat_hidden: int = 64,
        device: torch.device = torch.device("cpu"),
        prior_edge_prob: float = 0.3,
        prior_a_mu: float = 0.0,
        prior_a_std: float = 1.0,
        tau_init: float = 0.5,
    ):
        super().__init__()
        self.num_vars = num_vars
        self.lag = lag
        self.feat_dim = feat_dim
        self.device = device

        # Shared Deep-Kernel feature extractor
        self.feature_extractor = DeepFeatureExtractor(lag, feat_dim, feat_hidden)

        # ── Bayesian causal weight parameters ──
        # Edge gate z: q(z[i,j]) via Concrete relaxation
        # z_logit[i,j] → P(edge) = sigmoid(z_logit[i,j])
        init_logit = math.log(prior_edge_prob / (1.0 - prior_edge_prob))
        self.z_logit = nn.Parameter(
            torch.full((num_vars, num_vars), init_logit))

        # Edge strength a: q(a[i,j]) = Softplus-Normal
        # a = softplus(a_raw), a_raw ~ N(a_mu, exp(a_log_std)²)
        self.a_mu = nn.Parameter(torch.zeros(num_vars, num_vars))
        self.a_log_std = nn.Parameter(
            torch.full((num_vars, num_vars), math.log(0.5)))

        # Concrete temperature (annealed during training)
        self.register_buffer("tau", torch.tensor(tau_init))

        # Prior hyperparameters
        self.register_buffer("prior_z_prob",
                             torch.tensor(prior_edge_prob))
        self.register_buffer("prior_a_mu_val",
                             torch.tensor(prior_a_mu))
        self.register_buffer("prior_a_std_val",
                             torch.tensor(prior_a_std))

        # N per-target MKL GPs (each with N per-source kernels)
        self._models = nn.ModuleList([
            PerTargetMKL(num_vars) for _ in range(num_vars)
        ])

        self.fitted = False
        self._raw_feats = None   # {j: Tensor(n, lag)}
        self._targets = None     # {i: Tensor(n,)}

    # ── Sampling ──────────────────────────────────────────────────────

    def sample_causal_weights(self) -> torch.Tensor:
        """Sample effective causal weights W_eff = z * a.

        During training: Concrete relaxation for z, reparameterized a.
        During eval: E[W] = P(z=1) * E[a] — preserves posterior uncertainty
                     instead of hard-gating at 0.5.
        Returns (N, N) non-negative effective weights.
        """
        if self.training:
            # Concrete / Gumbel-sigmoid for z
            u = torch.rand_like(self.z_logit).clamp(1e-6, 1 - 1e-6)
            gumbel_noise = torch.log(u) - torch.log(1.0 - u)
            z = torch.sigmoid((self.z_logit + gumbel_noise) / self.tau)

            # Reparameterized a
            eps = torch.randn_like(self.a_mu)
            a_raw = self.a_mu + torch.exp(self.a_log_std) * eps
            a = F.softplus(a_raw)
        else:
            # Posterior mean: E[W] = P(z=1) * E[a]
            # Preserves uncertainty — P(z=1)=0.51 contributes less than P(z=1)=0.99
            z = torch.sigmoid(self.z_logit)
            a = F.softplus(self.a_mu)

        return z * a

    @property
    def causal_weights(self) -> torch.Tensor:
        """Effective causal weights (N, N): z * a.

        Differentiable during training (sampled), deterministic during eval.
        """
        return self.sample_causal_weights()

    @property
    def edge_probs(self) -> torch.Tensor:
        """Edge existence probabilities P(z=1) = sigmoid(z_logit). (N, N)"""
        return torch.sigmoid(self.z_logit)

    @property
    def edge_strength_mean(self) -> torch.Tensor:
        """Mean edge strength softplus(a_mu). (N, N)"""
        return F.softplus(self.a_mu)

    @property
    def edge_strength_std(self) -> torch.Tensor:
        """Approximate std of edge strength. (N, N)"""
        return torch.exp(self.a_log_std)

    # ── KL Divergence ──────────────────────────────────────────────────

    def kl_divergence(self) -> torch.Tensor:
        """KL(q(z,a) || p(z,a)) = KL_z + KL_a.

        KL_z: KL(Bernoulli(sigmoid(z_logit)) || Bernoulli(prior_z_prob))
        KL_a: KL(N(a_mu, a_std²) || N(prior_a_mu, prior_a_std²))
        """
        # KL for edge gates (Bernoulli approximation of Concrete)
        p = torch.sigmoid(self.z_logit)
        p = p.clamp(1e-6, 1 - 1e-6)
        q0 = self.prior_z_prob.clamp(1e-6, 1 - 1e-6)
        kl_z = (p * (torch.log(p) - torch.log(q0))
                + (1 - p) * (torch.log(1 - p) - torch.log(1 - q0)))
        kl_z = kl_z.sum()

        # KL for edge strengths (Gaussian)
        a_var = torch.exp(2.0 * self.a_log_std)
        prior_var = self.prior_a_std_val ** 2
        kl_a = (0.5 * (torch.log(prior_var / a_var)
                        + (a_var + (self.a_mu - self.prior_a_mu_val) ** 2) / prior_var
                        - 1.0))
        kl_a = kl_a.sum()

        return kl_z + kl_a

    # ── Causal summary for evaluation ─────────────────────────────────

    def causal_scores(self) -> torch.Tensor:
        """Continuous causal scores for AUROC/AUPRC: P(z=1) * E[a]. (N, N)"""
        return (self.edge_probs * self.edge_strength_mean).detach()

    def causal_summary(self) -> Dict[str, torch.Tensor]:
        """Full Bayesian causal summary."""
        return {
            "edge_prob": self.edge_probs.detach().cpu(),
            "strength_mean": self.edge_strength_mean.detach().cpu(),
            "strength_std": self.edge_strength_std.detach().cpu(),
            "effective": self.causal_scores().cpu(),
        }

    def _get_model(self, i: int, j: int = 0) -> PerTargetMKL:
        return self._models[i]

    # ------------------------------------------------------------------
    #  Feature extraction
    # ------------------------------------------------------------------
    def _extract_source_features(
        self, raw_feats: dict, differentiable: bool = True,
    ) -> List[torch.Tensor]:
        """Extract deep features per source using shared MLP.
        Returns list of (m, d_feat)."""
        N = self.num_vars
        if differentiable:
            return [self.feature_extractor(raw_feats[j]) for j in range(N)]
        else:
            with torch.no_grad():
                return [self.feature_extractor(raw_feats[j]) for j in range(N)]

    # ------------------------------------------------------------------
    #  Build feature / target pairs
    # ------------------------------------------------------------------
    def _build_pairs(self, X_seq: torch.Tensor):
        """X_seq: (T, N).  Returns raw_feats[j] and targets[i]."""
        T, N = X_seq.shape
        raw_feats = {}
        targets = {}
        for j in range(N):
            feats = []
            for t in range(self.lag, T):
                feats.append(X_seq[t - self.lag:t, j].flip(0))
            raw_feats[j] = torch.stack(feats, dim=0)       # (m, lag)
        for i in range(N):
            targets[i] = X_seq[self.lag:, i]                # (m,)
        return raw_feats, targets

    # ------------------------------------------------------------------
    #  Fit (ELBO optimisation: E_q[MLL] - KL(q||p))
    # ------------------------------------------------------------------
    def fit(
        self,
        normal_windows,
        iters: int = 50,
        lr: float = 0.05,
        max_train_samples: int = 500,
        beta_w: float = 1.0,
        tau_anneal_start: float = 1.0,
        tau_anneal_end: float = 0.3,
    ):
        """Jointly optimise feature extractors + Bayesian causal weights + GPs.

        Loss = -E_q[MLL]/N + beta_w * KL(q(z,a) || p(z,a)) / N²
        Temperature τ is annealed from tau_anneal_start to tau_anneal_end.
        """
        import numpy as np
        from tqdm import tqdm

        # ── Concatenate normal sequences ──
        seqs = []
        total_len = 0
        for arr in normal_windows:
            if arr.ndim == 3:
                arr = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])
            seqs.append(arr)
            total_len += arr.shape[0]
            if total_len > 20000:
                break

        X_all = torch.tensor(
            np.concatenate(seqs, axis=0), dtype=torch.float, device=self.device)
        raw_feats, targets = self._build_pairs(X_all)

        # ── Sub-sample ──
        N = self.num_vars
        n_samples = raw_feats[0].shape[0]
        if n_samples > max_train_samples:
            idx = torch.randperm(n_samples, device=self.device)[:max_train_samples]
            for j in range(N):
                raw_feats[j] = raw_feats[j][idx]
            for i in range(N):
                targets[i] = targets[i][idx]

        self._raw_feats = raw_feats
        self._targets = targets

        # ── Joint optimisation ──
        self.to(self.device)
        self.train()
        opt = torch.optim.Adam(self.parameters(), lr=lr)

        pbar = tqdm(range(iters), desc="  Causal GP ELBO")
        for it in pbar:
            # Anneal temperature
            frac = it / max(1, iters - 1)
            self.tau.fill_(
                tau_anneal_start + (tau_anneal_end - tau_anneal_start) * frac)

            opt.zero_grad()

            Z_lists = self._extract_source_features(raw_feats, differentiable=True)
            W = self.sample_causal_weights()  # (N, N), sampled

            total_mll = torch.tensor(0.0, device=self.device)
            for i in range(N):
                total_mll = total_mll + self._models[i].mll(
                    Z_lists, targets[i], W[i]).squeeze()

            kl_w = self.kl_divergence()
            # Normalise KL by number of edges (N²)
            loss = -total_mll / N + beta_w * kl_w / (N * N)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
            opt.step()

            if (it + 1) % max(1, iters // 5) == 0 or it == iters - 1:
                p_mean = self.edge_probs.mean().item()
                pbar.set_postfix({
                    "mll": f"{total_mll.item()/N:.4f}",
                    "kl": f"{kl_w.item()/(N*N):.4f}",
                    "P(e)": f"{p_mean:.3f}",
                    "τ": f"{self.tau.item():.3f}",
                })

        # ── Cache posteriors (use mean weights for inference) ──
        self.eval()
        with torch.no_grad():
            Z_lists = self._extract_source_features(raw_feats, differentiable=False)
            W = self.causal_weights  # deterministic in eval mode
            for i in range(N):
                self._models[i].cache_posterior(Z_lists, targets[i], W[i])

        self.fitted = True

    # ------------------------------------------------------------------
    #  Re-cache posteriors (for end-to-end training)
    # ------------------------------------------------------------------
    def refit_posteriors(self, refit_kernel_iters: int = 0, lr: float = 0.05):
        assert self._raw_feats is not None, "Must call fit() first"
        with torch.no_grad():
            Z_lists = self._extract_source_features(self._raw_feats, differentiable=False)
            W = self.causal_weights
            for i in range(self.num_vars):
                self._models[i].cache_posterior(Z_lists, self._targets[i], W[i])

    # ------------------------------------------------------------------
    #  Prediction
    # ------------------------------------------------------------------
    def predict_sequence(
        self,
        X_seq: torch.Tensor,
        differentiable: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict mu, sigma for all valid time steps.

        Returns:
            mu:    (m, N)  per-target prediction means  (m = T - lag)
            sigma: (m, N)  per-target prediction std devs
        """
        assert self.fitted
        T, N = X_seq.shape
        m = T - self.lag

        raw_feats = {}
        for j in range(N):
            feats = []
            for t in range(self.lag, T):
                feats.append(X_seq[t - self.lag:t, j].flip(0))
            raw_feats[j] = torch.stack(feats, dim=0)

        Z_lists = self._extract_source_features(raw_feats, differentiable=differentiable)
        W = self.causal_weights  # (N, N)

        mu = torch.zeros(m, N, device=self.device)
        sigma = torch.zeros(m, N, device=self.device)

        for i in range(N):
            mean_pred, var_pred = self._models[i].predict(Z_lists, W[i])

            if differentiable:
                mu[:, i] = mean_pred
                sigma[:, i] = var_pred.sqrt()
            else:
                mu[:, i] = mean_pred.detach()
                sigma[:, i] = var_pred.sqrt().detach()

        return mu, sigma

    def predict_sequence_differentiable(
        self,
        X_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fully differentiable prediction for end-to-end training.

        Recomputes GP posteriors from scratch using sampled Bayesian
        causal weights.  Gradients flow through:
          - test features  -> feature extractor
          - train features -> feature extractor
          - K_train        -> z, a (causal weights)
          - alpha, L       -> K_train -> z, a, feature extractor

        Returns total MLL, KL divergence, and the sampled W used for GP
        predictions (so the caller can reuse the SAME W for VGAE adjacency,
        avoiding the double-sampling inconsistency).

        Args:
            X_seq: (T, N) input sequence
        Returns:
            mu: (m, N), sigma: (m, N), total_mll: scalar, kl_w: scalar,
            W: (N, N) sampled causal weights used in this forward pass
        """
        assert self._raw_feats is not None, "Must call fit() first"
        T, N = X_seq.shape
        m = T - self.lag

        # Build test raw features
        raw_feats_test = {}
        for j in range(N):
            feats = []
            for t in range(self.lag, T):
                feats.append(X_seq[t - self.lag:t, j].flip(0))
            raw_feats_test[j] = torch.stack(feats, dim=0)

        # Extract features differentiably (both train and test use current g)
        Z_train = self._extract_source_features(self._raw_feats, differentiable=True)
        Z_test = self._extract_source_features(raw_feats_test, differentiable=True)

        W = self.sample_causal_weights()  # (N, N), sampled ONCE
        kl_w = self.kl_divergence()

        mu = torch.zeros(m, N, device=self.device)
        sigma = torch.zeros(m, N, device=self.device)
        total_mll = torch.tensor(0.0, device=self.device)

        for i in range(N):
            mean_pred, var_pred, mll_i = self._models[i].predict_differentiable(
                Z_train, self._targets[i], Z_test, W[i])
            mu[:, i] = mean_pred
            sigma[:, i] = var_pred.sqrt()
            total_mll = total_mll + mll_i

        return mu, sigma, total_mll, kl_w, W

    @torch.no_grad()
    def infer_single(self, X_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict for the LAST valid time step only.

        X_seq: (T, N),  T >= lag
        Returns mu: (N,), sigma: (N,)
        """
        assert self.fitted
        T, N = X_seq.shape
        W = self.causal_weights

        Z_sources = []
        for j in range(N):
            xj = X_seq[T - self.lag:T, j].flip(0).view(1, -1)
            zj = self.feature_extractor(xj)    # (1, d_feat)
            Z_sources.append(zj)

        mu = torch.zeros(N, device=self.device)
        sigma = torch.zeros(N, device=self.device)

        for i in range(N):
            mean, var = self._models[i].predict(Z_sources, W[i])
            mu[i] = mean.item()
            sigma[i] = var.sqrt().item()

        return mu, sigma

