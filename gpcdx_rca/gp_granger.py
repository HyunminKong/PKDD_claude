from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


# ======================================================================
#  Deep Feature Extractor (Deep Kernel Learning)
# ======================================================================

class DeepFeatureExtractor(nn.Module):
    """MLP feature extractor for Deep Kernel Learning.

    Maps raw lagged time-series window R^P -> R^d_feat so that the RBF
    kernel operates in a learned low-dimensional space instead of the
    raw (potentially very high-dimensional) lag space.

    Shared across all N^2 pairwise GPs for parameter efficiency.
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
    """RBF kernel: k(x,x') = sigma_f^2 exp(-1/(2 ell^2) ||x-x'||^2)"""

    def __init__(self, lengthscale: float = 1.0, variance: float = 1.0):
        super().__init__()
        self.log_lengthscale = nn.Parameter(
            torch.tensor(math.log(lengthscale), dtype=torch.float))
        self.log_variance = nn.Parameter(
            torch.tensor(math.log(variance), dtype=torch.float))

    def forward(self, X: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        ls2 = torch.exp(self.log_lengthscale) ** 2
        var = torch.exp(self.log_variance)
        # X: (n, d), Z: (m, d)
        X2 = (X ** 2).sum(-1, keepdim=True)
        Z2 = (Z ** 2).sum(-1, keepdim=True).transpose(0, 1)
        cross = X @ Z.transpose(0, 1)
        dist2 = X2 + Z2 - 2.0 * cross
        return var * torch.exp(-0.5 * dist2 / ls2.clamp(min=1e-6))


# ======================================================================
#  Per-pair GP (operates in feature space)
# ======================================================================

class PairwiseDeepGP(nn.Module):
    """Single GP with per-pair kernel hyperparameters.

    Uses an external (shared) feature extractor.  All kernel computations
    happen in the learned feature space R^d_feat.

    Supports **differentiable** prediction: when the cached posterior
    (Z_train, alpha, L) is fixed, gradients still flow through the
    test-point features z_star, enabling end-to-end training.
    """

    def __init__(self):
        super().__init__()
        self.kernel = RBFKernel()
        self.log_noise = nn.Parameter(
            torch.tensor(math.log(1e-2), dtype=torch.float))
        # Cached posterior (populated by cache_posterior)
        self.Z_train = None   # (n, d_feat) — detached
        self.alpha = None      # (n, 1)      — detached
        self.L = None          # (n, n)      — detached

    # ---- Robust Cholesky with adaptive jitter ----
    @staticmethod
    def _robust_cholesky(M: torch.Tensor, max_tries: int = 6) -> torch.Tensor:
        """Cholesky with exponentially increasing jitter on failure."""
        jitter = 1e-6
        for _ in range(max_tries):
            try:
                return torch.linalg.cholesky(
                    M + jitter * torch.eye(M.shape[0], dtype=M.dtype, device=M.device))
            except RuntimeError:
                jitter *= 10.0
        # Last resort: large jitter
        return torch.linalg.cholesky(
            M + 1e-1 * torch.eye(M.shape[0], dtype=M.dtype, device=M.device))

    # ---- MLL (Eq 16) ----
    def mll(self, K: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n = y.shape[0]
        noise = torch.exp(self.log_noise)
        L = self._robust_cholesky(K + noise * torch.eye(n, dtype=K.dtype, device=K.device))
        alpha = torch.cholesky_solve(y.view(-1, 1), L)
        log_det = 2.0 * torch.log(torch.diag(L)).sum()
        return (-0.5 * y.view(1, -1) @ alpha
                - 0.5 * log_det
                - 0.5 * n * math.log(2 * math.pi))

    # ---- Cache posterior for fast (differentiable) prediction ----
    def cache_posterior(self, Z: torch.Tensor, y: torch.Tensor):
        """Store Z_train, alpha, L (all detached) for prediction."""
        with torch.no_grad():
            K = self.kernel(Z, Z)
            n = Z.shape[0]
            noise = torch.exp(self.log_noise)
            L = self._robust_cholesky(K + noise * torch.eye(n, dtype=K.dtype, device=K.device))
            self.Z_train = Z.detach().clone()
            self.L = L.detach()
            self.alpha = torch.cholesky_solve(y.view(-1, 1), L).detach()

    # ---- Prediction (differentiable through z_star) ----
    def predict(self, z_star: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict mean and variance.

        Gradients flow through *z_star* (feature-extracted test input).
        The cached Z_train / alpha / L are treated as constants.

        Args:
            z_star: (m, d_feat)
        Returns:
            mean: (m,)
            var:  (m,)   (clamped >= 1e-9)
        """
        Kx = self.kernel(z_star, self.Z_train)            # (m, n)
        mean = (Kx @ self.alpha).view(-1)
        v = torch.linalg.solve_triangular(
            self.L, Kx.transpose(0, 1), upper=False)       # (n, m)
        Kss = self.kernel(z_star, z_star).diagonal()
        var = Kss - (v ** 2).sum(dim=0) + torch.exp(self.log_noise)
        return mean, var.clamp(min=1e-9)


# ======================================================================
#  Granger-GP-VAR with Deep Kernel (Section 4.2 — enhanced)
# ======================================================================

class GrangerGPVAR(nn.Module):
    """Distributional Granger causality via N^2 Deep-Kernel GPs.

    For each pair (i <- j) an independent GP maps:
        g_theta(x_{t-P:t}^{(j)})  in R^d_feat   -->   mu_{ij}(t), sigma_{ij}(t)

    where g_theta is a **shared** neural feature extractor (Deep Kernel
    Learning) and the RBF kernel + noise are **per-pair** hyperparameters.

    Key improvement over vanilla GP-VAR:
      * The shared feature extractor compresses R^P -> R^d_feat, avoiding
        the curse of dimensionality for large P.
      * Predictions are differentiable w.r.t. g_theta, enabling end-to-end
        training with the downstream VGAE.
    """

    def __init__(
        self,
        num_vars: int,
        lag: int,
        feat_dim: int = 16,
        feat_hidden: int = 64,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.num_vars = num_vars
        self.lag = lag
        self.feat_dim = feat_dim
        self.device = device

        # Shared Deep-Kernel feature extractor
        self.feature_extractor = DeepFeatureExtractor(lag, feat_dim, feat_hidden)

        # N^2 independent per-pair GPs (own kernel hyper-params each)
        self._models = nn.ModuleList([
            PairwiseDeepGP() for _ in range(num_vars * num_vars)
        ])

        self.fitted = False
        # Raw training data (kept for posterior re-caching)
        self._raw_feats = None   # {j: Tensor(n, lag)}
        self._targets = None     # {(i,j): Tensor(n,)}

    def _get_model(self, i: int, j: int) -> PairwiseDeepGP:
        return self._models[i * self.num_vars + j]

    # ------------------------------------------------------------------
    #  Build feature / target pairs
    # ------------------------------------------------------------------
    def _build_pairs(self, X_seq: torch.Tensor):
        """X_seq: (T, N).  Returns raw_feats[j] and targets[(i,j)]."""
        T, N = X_seq.shape
        raw_feats = {}
        targets = {}
        for j in range(N):
            feats = []
            for t in range(self.lag, T):
                feats.append(X_seq[t - self.lag:t, j].flip(0))
            raw_feats[j] = torch.stack(feats, dim=0)          # (m, lag)
            for i in range(N):
                targets[(i, j)] = X_seq[self.lag:, i]         # (m,)
        return raw_feats, targets

    # ------------------------------------------------------------------
    #  Fit (joint MLL optimisation of feature extractor + all kernels)
    # ------------------------------------------------------------------
    def fit(
        self,
        normal_windows,
        iters: int = 50,
        lr: float = 0.05,
        max_train_samples: int = 500,
    ):
        """Jointly optimise shared feature extractor + N^2 GP kernels via MLL."""
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

        # ── Sub-sample per source for efficiency ──
        N = self.num_vars
        for j in range(N):
            n_j = raw_feats[j].shape[0]
            if n_j > max_train_samples:
                idx = torch.randperm(n_j, device=self.device)[:max_train_samples]
                raw_feats[j] = raw_feats[j][idx]
                for i in range(N):
                    targets[(i, j)] = targets[(i, j)][idx]

        # Store for later re-fitting
        self._raw_feats = raw_feats
        self._targets = targets

        # ── Joint optimisation ──
        self.to(self.device)
        opt = torch.optim.Adam(self.parameters(), lr=lr)

        pbar = tqdm(range(iters), desc="  Deep-Kernel GP MLL")
        for it in pbar:
            opt.zero_grad()
            total_mll = torch.tensor(0.0, device=self.device)

            for j in range(N):
                Zj = self.feature_extractor(raw_feats[j])     # shared extraction
                for i in range(N):
                    yi = targets[(i, j)]
                    K = self._get_model(i, j).kernel(Zj, Zj)
                    total_mll = total_mll + self._get_model(i, j).mll(K, yi).squeeze()

            loss = -total_mll / (N * N)
            loss.backward()
            opt.step()

            if (it + 1) % max(1, iters // 5) == 0 or it == iters - 1:
                pbar.set_postfix({"avg_mll": f"{-loss.item():.4f}"})

        # ── Cache posteriors ──
        with torch.no_grad():
            for j in range(N):
                Zj = self.feature_extractor(raw_feats[j])
                for i in range(N):
                    self._get_model(i, j).cache_posterior(Zj, targets[(i, j)])

        self.fitted = True

    # ------------------------------------------------------------------
    #  Re-cache posteriors (for end-to-end training)
    # ------------------------------------------------------------------
    def refit_posteriors(self, refit_kernel_iters: int = 0, lr: float = 0.05):
        """Re-cache GP posteriors using the current feature extractor.

        Optionally fine-tune kernel hyper-parameters for a few MLL steps
        (feature extractor is frozen during this phase).
        """
        assert self._raw_feats is not None, "Must call fit() first"
        N = self.num_vars

        if refit_kernel_iters > 0:
            # Only optimise per-pair kernel params (not feature extractor)
            kernel_params = []
            for m in self._models:
                kernel_params.extend([m.kernel.log_lengthscale,
                                      m.kernel.log_variance,
                                      m.log_noise])
            opt = torch.optim.Adam(kernel_params, lr=lr)

            for _ in range(refit_kernel_iters):
                opt.zero_grad()
                total_mll = torch.tensor(0.0, device=self.device)
                with torch.no_grad():
                    feats_j = {j: self.feature_extractor(self._raw_feats[j])
                               for j in range(N)}
                for j in range(N):
                    Zj = feats_j[j]
                    for i in range(N):
                        K = self._get_model(i, j).kernel(Zj, Zj)
                        total_mll = total_mll + self._get_model(
                            i, j).mll(K, self._targets[(i, j)]).squeeze()
                (-total_mll / (N * N)).backward()
                opt.step()

        # Re-cache with updated features
        with torch.no_grad():
            for j in range(N):
                Zj = self.feature_extractor(self._raw_feats[j])
                for i in range(N):
                    self._get_model(i, j).cache_posterior(
                        Zj, self._targets[(i, j)])

    # ------------------------------------------------------------------
    #  Prediction
    # ------------------------------------------------------------------
    def predict_sequence(
        self,
        X_seq: torch.Tensor,
        differentiable: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch-predict mu_{ij}(t), sigma_{ij}(t) for all valid time steps.

        Args:
            X_seq: (T, N) time series
            differentiable: if True, gradients flow through the feature
                extractor (for end-to-end training).

        Returns:
            mu:    (m, N, N)  where m = T - lag
            sigma: (m, N, N)  standard deviations
        """
        assert self.fitted
        T, N = X_seq.shape
        m = T - self.lag

        mu = torch.zeros(m, N, N, device=self.device)
        sigma = torch.zeros(m, N, N, device=self.device)

        for j in range(N):
            # Build lagged feature matrix for source j
            feats = []
            for k in range(m):
                t = self.lag + k
                feats.append(X_seq[t - self.lag:t, j].flip(0))
            Xj_raw = torch.stack(feats, dim=0)                # (m, lag)

            # Feature extraction
            if differentiable:
                Zj = self.feature_extractor(Xj_raw)
            else:
                with torch.no_grad():
                    Zj = self.feature_extractor(Xj_raw)

            for i in range(N):
                model = self._get_model(i, j)
                mean_pred, var_pred = model.predict(Zj)

                if differentiable:
                    mu[:, i, j] = mean_pred
                    sigma[:, i, j] = var_pred.sqrt()
                else:
                    mu[:, i, j] = mean_pred.detach()
                    sigma[:, i, j] = var_pred.sqrt().detach()

        return mu, sigma

    @torch.no_grad()
    def infer_single(self, X_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict mu, sigma for the LAST valid time step only.

        X_seq: (T, N),  T >= lag
        Returns mu: (N, N), sigma: (N, N)
        """
        assert self.fitted
        T, N = X_seq.shape
        mu = torch.zeros(N, N, device=self.device)
        sigma = torch.zeros(N, N, device=self.device)
        for j in range(N):
            xj = X_seq[T - self.lag:T, j].flip(0).view(1, -1)   # (1, lag)
            zj = self.feature_extractor(xj)                       # (1, d_feat)
            for i in range(N):
                model = self._get_model(i, j)
                mean, var = model.predict(zj)
                mu[i, j] = mean.item()
                sigma[i, j] = var.sqrt().item()
        return mu, sigma
