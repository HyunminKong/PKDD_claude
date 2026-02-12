from __future__ import annotations

import time
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, hamming_loss,
)
from tqdm import tqdm

from .gp_granger import GrangerGPVAR
from .vgae import VGAE
from .utils import normalize_adjacency


class GPCDX_RCA:
    """GPCDX for Root Cause Analysis — enhanced with Deep Kernel Learning.

    Architecture:
      Stage 1: GrangerGPVAR — N^2 Deep-Kernel GPs (shared feature extractor)
      Stage 2: ExogenousVGAE — GCN-based VAE

    Training modes:
      * two_stage:   Fit GP via MLL -> pre-compute predictions -> train VGAE
      * joint:       Fit GP via MLL -> train VGAE with on-the-fly GP prediction
      * end_to_end:  Fit GP via MLL (warm-up) -> jointly optimise feature
                     extractor + VGAE with differentiable GP predictions

    RCA Scoring:
      Per-node scores = Granger W2 + Exogenous W2 + Reconstruction MSE
    """

    def __init__(
        self,
        num_vars: int,
        window_size: int,
        device: str = "cpu",
        gp_iters: int = 30,
        gp_lr: float = 0.05,
        gp_max_train: int = 500,
        feat_dim: int = 16,
        feat_hidden: int = 64,
        vgae_hidden: int = 64,
        vgae_latent: int = 8,
        epochs: int = 10,
        batch_size: int = 256,
        lr: float = 1e-3,
        beta_kl: float = 1.0,
        sparse_weight: float = 0.01,
    ):
        self.N = num_vars
        self.P = window_size
        self.device = torch.device(device)

        # Deep-Kernel GP (lag = P)
        self.gp = GrangerGPVAR(
            num_vars=num_vars, lag=window_size,
            feat_dim=feat_dim, feat_hidden=feat_hidden,
            device=self.device,
        )
        self.gp_iters = gp_iters
        self.gp_lr = gp_lr
        self.gp_max_train = gp_max_train

        # VGAE
        self.vgae = VGAE(
            P=self.P, N=num_vars, d=vgae_latent, hidden=vgae_hidden,
        ).to(self.device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.beta_kl = beta_kl
        self.sparse_weight = sparse_weight

        # Reference distributions (set during training)
        self.A_norm = None
        self.mu_ref = None
        self.sigma_ref = None
        self.exo_mu_ref = None
        self.exo_sigma_ref = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _prepare_vgae_samples(
        self,
        X_seq: torch.Tensor,
        mu_seq: torch.Tensor,
        sigma_seq: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], ...]:
        """Extract per-step VGAE inputs from one sequence.

        Preserves gradient graph when mu_seq / sigma_seq are differentiable.
        """
        P = self.P
        h_ts, mu_incs, x_ts, mu_sums, log_s_sums = [], [], [], [], []

        for k in range(mu_seq.shape[0]):
            t = P + k
            if t >= X_seq.shape[0]:
                break

            h_t = X_seq[t - P:t, :].T                             # (N, P)
            x_t = X_seq[t]                                         # (N,)
            mu_incoming = mu_seq[k].T                              # (N, N)

            # Summary stats (Eq 17)
            k_start = max(0, k - P + 1)
            mu_window = mu_seq[k_start:k + 1]
            sigma_window = sigma_seq[k_start:k + 1]

            mu_sum = mu_window.sum(dim=(0, 1))                     # (N,)
            log_sigma_sum = torch.log(
                sigma_window.pow(2).sum(dim=(0, 1)).clamp(min=1e-9))

            h_ts.append(h_t)
            mu_incs.append(mu_incoming)
            x_ts.append(x_t)
            mu_sums.append(mu_sum)
            log_s_sums.append(log_sigma_sum)

        return h_ts, mu_incs, x_ts, mu_sums, log_s_sums

    def _update_granger_references(self, normal_windows: list):
        """Re-compute Granger references and adjacency from current GP."""
        mu_list, sigma_list = [], []
        with torch.no_grad():
            for arr in normal_windows:
                if arr.ndim == 3:
                    arr = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])
                X_seq = torch.tensor(arr, dtype=torch.float, device=self.device)
                if X_seq.shape[0] <= self.gp.lag:
                    continue
                mu_seq, sigma_seq = self.gp.predict_sequence(X_seq)
                mu_list.append(mu_seq)
                sigma_list.append(sigma_seq)

        if mu_list:
            mu_cat = torch.cat(mu_list, dim=0)
            sigma_cat = torch.cat(sigma_list, dim=0)
            self.mu_ref = mu_cat.mean(dim=0).to(self.device)
            self.sigma_ref = sigma_cat.mean(dim=0).to(self.device)
            self.A_norm = normalize_adjacency(self.mu_ref.abs()).to(self.device)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, normal_windows: list, training_mode: str = "two_stage"):
        if training_mode == "two_stage":
            self._fit_two_stage(normal_windows)
        elif training_mode == "joint":
            self._fit_joint(normal_windows)
        elif training_mode == "end_to_end":
            self._fit_end_to_end(normal_windows)
        else:
            raise ValueError(f"Unknown training_mode: {training_mode!r}")

    # ---- two-stage ---------------------------------------------------
    def _fit_two_stage(self, normal_windows: list):
        """Stage 1: pre-compute GP -> Stage 2: train VGAE from DataLoader."""
        # === Stage 1: Fit Deep-Kernel GP models ===
        print("=== Stage 1: Fitting Deep-Kernel GP models ===")
        self.gp.fit(normal_windows, iters=self.gp_iters, lr=self.gp_lr,
                     max_train_samples=self.gp_max_train)

        # Pre-compute GP predictions on all training sequences
        print("  Pre-computing GP predictions ...")
        all_h_t: List[torch.Tensor] = []
        all_mu_inc: List[torch.Tensor] = []
        all_x_t: List[torch.Tensor] = []
        all_mu_sum: List[torch.Tensor] = []
        all_log_s: List[torch.Tensor] = []
        all_mu: List[torch.Tensor] = []
        all_sigma: List[torch.Tensor] = []

        for arr in tqdm(normal_windows, desc="  GP predict"):
            if arr.ndim == 3:
                arr = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])
            X_seq = torch.tensor(arr, dtype=torch.float, device=self.device)
            if X_seq.shape[0] <= self.gp.lag:
                continue

            mu_seq, sigma_seq = self.gp.predict_sequence(X_seq)
            all_mu.append(mu_seq)
            all_sigma.append(sigma_seq)

            h_ts, mu_incs, x_ts, mu_sums, log_sigs = self._prepare_vgae_samples(
                X_seq, mu_seq, sigma_seq)
            all_h_t.extend(h_ts)
            all_mu_inc.extend(mu_incs)
            all_x_t.extend(x_ts)
            all_mu_sum.extend(mu_sums)
            all_log_s.extend(log_sigs)

        if not all_mu:
            raise ValueError("No valid training data after GP prediction")

        # Reference Granger distribution
        mu_cat = torch.cat(all_mu, dim=0)
        sigma_cat = torch.cat(all_sigma, dim=0)
        self.mu_ref = mu_cat.mean(dim=0).to(self.device)
        self.sigma_ref = sigma_cat.mean(dim=0).to(self.device)
        self.A_norm = normalize_adjacency(self.mu_ref.abs()).to(self.device)

        # Stack VGAE training tensors
        h_t_all = torch.stack(all_h_t).to(self.device)
        mu_inc_all = torch.stack(all_mu_inc).to(self.device)
        x_t_all = torch.stack(all_x_t).to(self.device)
        mu_sum_all = torch.stack(all_mu_sum).to(self.device)
        ls_all = torch.stack(all_log_s).to(self.device)

        X_enc_all = torch.cat(
            [h_t_all, mu_inc_all, x_t_all.unsqueeze(-1)], dim=-1)

        print(f"  VGAE training samples: {X_enc_all.shape[0]}")

        # === Stage 2: Train VGAE ===
        print("=== Stage 2: Training VGAE ===")
        ds = TensorDataset(X_enc_all, h_t_all, x_t_all, mu_sum_all, ls_all)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        opt = torch.optim.Adam(self.vgae.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            self.vgae.train()
            total_rec, total_kl, n_batch = 0.0, 0.0, 0
            pbar = tqdm(loader, desc=f"  VGAE epoch {epoch+1}/{self.epochs}")
            for X_enc_b, h_t_b, x_t_b, ms_b, ls_b in pbar:
                x_hat, mu_u, logvar_u = self.vgae(
                    X_enc_b, self.A_norm, h_t_b, ms_b, ls_b)
                recon = F.mse_loss(x_hat, x_t_b)
                kl = -0.5 * torch.mean(
                    torch.sum(1 + logvar_u - mu_u.pow(2) - logvar_u.exp(),
                              dim=-1))
                loss = recon + self.beta_kl * kl
                opt.zero_grad()
                loss.backward()
                opt.step()

                total_rec += recon.item()
                total_kl += kl.item()
                n_batch += 1
                pbar.set_postfix({
                    "rec": f"{total_rec / n_batch:.4f}",
                    "kl": f"{total_kl / n_batch:.4f}",
                })

        # Reference exogenous distributions
        print("  Computing reference exogenous distributions ...")
        self._compute_exo_reference(X_enc_all, h_t_all, mu_sum_all, ls_all)
        print("  Training complete.")

    # ---- joint -------------------------------------------------------
    def _fit_joint(self, normal_windows: list):
        """GP fitting + VGAE training (on-the-fly GP, no differentiable)."""
        print("=== Fitting Deep-Kernel GP models ===")
        self.gp.fit(normal_windows, iters=self.gp_iters, lr=self.gp_lr,
                     max_train_samples=self.gp_max_train)

        print("  Computing Granger references ...")
        self._update_granger_references(normal_windows)

        print("=== Training VGAE (joint mode) ===")
        opt = torch.optim.Adam(self.vgae.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            self.vgae.train()
            total_rec, total_kl, n_batch = 0.0, 0.0, 0
            pbar = tqdm(normal_windows, desc=f"  epoch {epoch+1}/{self.epochs}")

            for arr in pbar:
                if arr.ndim == 3:
                    arr = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])
                X_seq = torch.tensor(arr, dtype=torch.float, device=self.device)
                if X_seq.shape[0] <= self.gp.lag:
                    continue

                mu_seq, sigma_seq = self.gp.predict_sequence(X_seq)
                h_ts, mu_incs, x_ts, mu_sums, log_sigs = \
                    self._prepare_vgae_samples(X_seq, mu_seq, sigma_seq)

                if not h_ts:
                    continue

                h_t_b = torch.stack(h_ts)
                mu_inc_b = torch.stack(mu_incs)
                x_t_b = torch.stack(x_ts)
                ms_b = torch.stack(mu_sums)
                ls_b = torch.stack(log_sigs)

                X_enc_b = torch.cat(
                    [h_t_b, mu_inc_b, x_t_b.unsqueeze(-1)], dim=-1)

                n_samples = X_enc_b.shape[0]
                indices = torch.randperm(n_samples, device=self.device)

                for start in range(0, n_samples, self.batch_size):
                    end = min(start + self.batch_size, n_samples)
                    idx = indices[start:end]

                    x_hat, mu_u, logvar_u = self.vgae(
                        X_enc_b[idx], self.A_norm, h_t_b[idx], ms_b[idx], ls_b[idx])
                    recon = F.mse_loss(x_hat, x_t_b[idx])
                    kl = -0.5 * torch.mean(
                        torch.sum(1 + logvar_u - mu_u.pow(2) - logvar_u.exp(),
                                  dim=-1))
                    loss = recon + self.beta_kl * kl
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    total_rec += recon.item()
                    total_kl += kl.item()
                    n_batch += 1

                pbar.set_postfix({
                    "rec": f"{total_rec / max(n_batch, 1):.4f}",
                    "kl": f"{total_kl / max(n_batch, 1):.4f}",
                })

        print("  Computing reference exogenous distributions ...")
        self._compute_exo_reference_joint(normal_windows)
        print("  Training complete.")

    # ---- end-to-end --------------------------------------------------
    def _fit_end_to_end(self, normal_windows: list):
        """End-to-end: GP feature extractor + VGAE jointly optimised.

        Phase 1: Warm-up Deep-Kernel GP via MLL
        Phase 2: Joint training — VGAE loss back-propagates through
                 differentiable GP predictions to the feature extractor
        Phase 3: Final reference computation
        """
        # ═══ Phase 1: GP Warm-up ═══
        print("=== Phase 1: GP Warm-up (Deep Kernel MLL) ===")
        self.gp.fit(normal_windows, iters=self.gp_iters, lr=self.gp_lr,
                     max_train_samples=self.gp_max_train)

        print("  Computing initial Granger references ...")
        self._update_granger_references(normal_windows)

        # ═══ Phase 2: End-to-End Joint Training ═══
        print("=== Phase 2: End-to-End Training (differentiable A_norm + sparsity) ===")
        e2e_params = (
            list(self.gp.feature_extractor.parameters())
            + list(self.vgae.parameters())
        )
        opt = torch.optim.Adam(e2e_params, lr=self.lr)

        for epoch in range(self.epochs):
            self.vgae.train()
            self.gp.feature_extractor.train()
            total_rec, total_kl, total_sp, n_batch = 0.0, 0.0, 0.0, 0

            pbar = tqdm(normal_windows, desc=f"  e2e epoch {epoch+1}/{self.epochs}")
            for arr in pbar:
                if arr.ndim == 3:
                    arr = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])
                X_seq = torch.tensor(arr, dtype=torch.float, device=self.device)
                if X_seq.shape[0] <= self.gp.lag:
                    continue

                # Limit sequence length to control memory
                T = X_seq.shape[0]
                max_chunk = 500
                if T > max_chunk:
                    start_idx = torch.randint(0, T - max_chunk, (1,)).item()
                    X_seq = X_seq[start_idx:start_idx + max_chunk]

                # Differentiable GP predictions
                mu_seq, sigma_seq = self.gp.predict_sequence(
                    X_seq, differentiable=True)

                # ── Differentiable adjacency from GP predictions ──
                A_causal = mu_seq.abs().mean(dim=0).clamp(max=10.0)  # (N, N)
                A_norm_batch = normalize_adjacency(A_causal)         # differentiable

                h_ts, mu_incs, x_ts, mu_sums, log_sigs = \
                    self._prepare_vgae_samples(X_seq, mu_seq, sigma_seq)

                if not h_ts:
                    continue

                h_t_b = torch.stack(h_ts)
                mu_inc_b = torch.stack(mu_incs)
                x_t_b = torch.stack(x_ts)
                ms_b = torch.stack(mu_sums)
                ls_b = torch.stack(log_sigs)

                X_enc_b = torch.cat(
                    [h_t_b, mu_inc_b, x_t_b.unsqueeze(-1)], dim=-1)

                # Sub-sample to batch_size to control memory
                n_samples = X_enc_b.shape[0]
                if n_samples > self.batch_size:
                    idx = torch.randperm(n_samples, device=self.device)[:self.batch_size]
                    X_enc_b = X_enc_b[idx]
                    h_t_b = h_t_b[idx]
                    x_t_b = x_t_b[idx]
                    ms_b = ms_b[idx]
                    ls_b = ls_b[idx]

                # ── VGAE forward with differentiable A_norm ──
                x_hat, mu_u, logvar_u = self.vgae(
                    X_enc_b, A_norm_batch, h_t_b, ms_b, ls_b)

                recon = F.mse_loss(x_hat, x_t_b)
                kl = -0.5 * torch.mean(
                    torch.sum(1 + logvar_u - mu_u.pow(2) - logvar_u.exp(),
                              dim=-1))
                # L1 sparsity on the causal adjacency
                sparsity = A_causal.mean()
                loss = recon + self.beta_kl * kl + self.sparse_weight * sparsity

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(e2e_params, max_norm=5.0)
                opt.step()

                total_rec += recon.item()
                total_kl += kl.item()
                total_sp += sparsity.item()
                n_batch += 1

                pbar.set_postfix({
                    "rec": f"{total_rec / max(n_batch, 1):.4f}",
                    "kl": f"{total_kl / max(n_batch, 1):.4f}",
                    "sp": f"{total_sp / max(n_batch, 1):.4f}",
                })

        # ═══ Phase 3: Final References ═══
        print("  Final GP posterior re-cache ...")
        self.gp.refit_posteriors(refit_kernel_iters=0)
        self._update_granger_references(normal_windows)
        print("  Computing reference exogenous distributions ...")
        self._compute_exo_reference_joint(normal_windows)
        print("  Training complete.")

    # ------------------------------------------------------------------
    # Exogenous reference computation
    # ------------------------------------------------------------------
    def _compute_exo_reference(self, X_enc, h_t, mu_sum, log_s):
        self.vgae.eval()
        all_mu_u, all_sigma_u = [], []
        BS = self.batch_size

        with torch.no_grad():
            for s in range(0, X_enc.shape[0], BS):
                e = min(s + BS, X_enc.shape[0])
                _, mu_u, logvar_u = self.vgae(
                    X_enc[s:e], self.A_norm, h_t[s:e], mu_sum[s:e], log_s[s:e])
                all_mu_u.append(mu_u)
                all_sigma_u.append(torch.exp(0.5 * logvar_u))

        all_mu_u = torch.cat(all_mu_u, dim=0)
        all_sigma_u = torch.cat(all_sigma_u, dim=0)
        self.exo_mu_ref = all_mu_u.mean(dim=0).to(self.device)
        self.exo_sigma_ref = all_sigma_u.mean(dim=0).to(self.device)

    def _compute_exo_reference_joint(self, normal_windows: list):
        self.vgae.eval()
        all_mu_u, all_sigma_u = [], []

        with torch.no_grad():
            for arr in normal_windows:
                if arr.ndim == 3:
                    arr = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])
                X_seq = torch.tensor(arr, dtype=torch.float, device=self.device)
                if X_seq.shape[0] <= self.gp.lag:
                    continue

                mu_seq, sigma_seq = self.gp.predict_sequence(X_seq)
                h_ts, mu_incs, x_ts, mu_sums, log_sigs = \
                    self._prepare_vgae_samples(X_seq, mu_seq, sigma_seq)

                if not h_ts:
                    continue

                h_t_b = torch.stack(h_ts)
                mu_inc_b = torch.stack(mu_incs)
                x_t_b = torch.stack(x_ts)
                ms_b = torch.stack(mu_sums)
                ls_b = torch.stack(log_sigs)

                X_enc_b = torch.cat(
                    [h_t_b, mu_inc_b, x_t_b.unsqueeze(-1)], dim=-1)

                for s in range(0, X_enc_b.shape[0], self.batch_size):
                    e = min(s + self.batch_size, X_enc_b.shape[0])
                    _, mu_u, logvar_u = self.vgae(
                        X_enc_b[s:e], self.A_norm, h_t_b[s:e], ms_b[s:e], ls_b[s:e])
                    all_mu_u.append(mu_u)
                    all_sigma_u.append(torch.exp(0.5 * logvar_u))

        all_mu_u = torch.cat(all_mu_u, dim=0)
        all_sigma_u = torch.cat(all_sigma_u, dim=0)
        self.exo_mu_ref = all_mu_u.mean(dim=0).to(self.device)
        self.exo_sigma_ref = all_sigma_u.mean(dim=0).to(self.device)

    # ------------------------------------------------------------------
    # RCA Scoring
    # ------------------------------------------------------------------
    @torch.no_grad()
    def rca_on_window(
        self, X_win: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute per-node RCA scores for a test window."""
        self.vgae.eval()
        X = torch.tensor(X_win, dtype=torch.float, device=self.device)
        T, N = X.shape

        mu_all, sigma_all = self.gp.predict_sequence(X)

        h_ts, mu_incs, x_ts, mu_sums, log_sigs = self._prepare_vgae_samples(
            X, mu_all, sigma_all)

        if not h_ts:
            z = torch.zeros(N, device=self.device)
            return z, z, z, z

        M = len(h_ts)
        h_t_b = torch.stack(h_ts)
        mu_inc_b = torch.stack(mu_incs)
        x_t_b = torch.stack(x_ts)
        ms_b = torch.stack(mu_sums)
        ls_b = torch.stack(log_sigs)

        X_enc = torch.cat(
            [h_t_b, mu_inc_b, x_t_b.unsqueeze(-1)], dim=-1)

        # Granger score
        start_k = max(0, self.P - self.gp.lag)
        mu_valid = mu_all[start_k:start_k + M]
        sigma_valid = sigma_all[start_k:start_k + M]

        Wg = (mu_valid - self.mu_ref) ** 2 + (sigma_valid - self.sigma_ref) ** 2
        granger_scores = Wg.sum(dim=1).mean(dim=0)

        # VGAE forward
        x_hat, mu_u, logvar_u = self.vgae(
            X_enc, self.A_norm, h_t_b, ms_b, ls_b)
        sigma_u = torch.exp(0.5 * logvar_u)

        # Exogenous score
        exo_d = ((mu_u - self.exo_mu_ref) ** 2
                 + (sigma_u - self.exo_sigma_ref) ** 2).sum(dim=-1)
        exo_scores = exo_d.mean(dim=0)

        # Reconstruction score
        recon_d = (x_hat - x_t_b) ** 2
        recon_scores = recon_d.mean(dim=0)

        node_scores = granger_scores + exo_scores + recon_scores
        return node_scores, granger_scores, exo_scores, recon_scores

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate_rca(
        self,
        x_ab_list,
        label_list,
        k_list=(1, 3, 5, 10),
    ) -> Dict[str, Any]:
        k_list = sorted(set(int(k) for k in k_list))
        max_k = max(k_list)
        hit_lists: Dict[int, list] = {k: [] for k in k_list}

        for idx in tqdm(range(len(x_ab_list)), desc="RCA eval"):
            Xw = x_ab_list[idx]
            if Xw.ndim == 3:
                Xw = Xw.squeeze(0) if Xw.shape[0] == 1 else Xw.reshape(-1, Xw.shape[-1])

            labels = label_list[idx]
            if labels.ndim == 1:
                true_vars = np.where(labels > 0)[0].tolist()
            else:
                true_vars = np.where(labels.max(axis=0) > 0)[0].tolist()

            if not true_vars:
                continue

            scores, _, _, _ = self.rca_on_window(Xw)
            topk_indices = torch.topk(
                scores, k=min(max_k, scores.numel())).indices.tolist()

            for k in k_list:
                kk = min(k, len(topk_indices))
                hit = 1.0 if any(v in topk_indices[:kk] for v in true_vars) else 0.0
                hit_lists[k].append(hit)

        metrics: Dict[str, Any] = {}
        for k in k_list:
            metrics[f"ac@{k}"] = float(np.mean(hit_lists[k])) if hit_lists[k] else 0.0
        metrics["n"] = len(hit_lists[k_list[0]]) if k_list else 0
        return metrics

    # ------------------------------------------------------------------
    # Causal Discovery Evaluation
    # ------------------------------------------------------------------
    def evaluate_causal_discovery(
        self,
        causal_struct: np.ndarray,
        test_sequences: list | None = None,
        causal_quantile: float = 0.70,
    ) -> Dict[str, float]:
        gt = causal_struct.astype(float).flatten()

        if test_sequences is not None and len(test_sequences) > 0:
            mu_ests = []
            for arr in test_sequences:
                if arr.ndim == 3:
                    arr = arr.reshape(-1, arr.shape[-1])
                X = torch.tensor(arr, dtype=torch.float, device=self.device)
                if X.shape[0] <= self.gp.lag:
                    continue
                mu_seq, _ = self.gp.predict_sequence(X)
                mu_ests.append(mu_seq.abs().mean(dim=0).cpu().numpy())
            if mu_ests:
                pred_cont = np.mean(mu_ests, axis=0)
            else:
                pred_cont = self.mu_ref.abs().cpu().numpy()
        else:
            pred_cont = self.mu_ref.abs().cpu().numpy()

        pred_flat = pred_cont.flatten()

        results: Dict[str, float] = {}
        try:
            results["auroc"] = float(roc_auc_score(gt, pred_flat))
        except ValueError:
            results["auroc"] = 0.0
        try:
            results["auprc"] = float(average_precision_score(gt, pred_flat))
        except ValueError:
            results["auprc"] = 0.0

        thr = float(np.quantile(pred_flat, causal_quantile))
        pred_bin = (pred_flat >= thr).astype(float)

        results["f1"] = float(f1_score(gt, pred_bin, zero_division=0))
        results["hamming"] = float(hamming_loss(gt, pred_bin))

        return results

    # ------------------------------------------------------------------
    # Model Complexity & Latency
    # ------------------------------------------------------------------
    def count_parameters(self) -> Dict[str, Any]:
        vgae_params = sum(p.numel() for p in self.vgae.parameters())
        vgae_trainable = sum(p.numel() for p in self.vgae.parameters() if p.requires_grad)

        # Feature extractor (shared)
        feat_params = sum(p.numel() for p in self.gp.feature_extractor.parameters())

        # Per-pair GP: kernel (2 params) + noise (1 param)
        gp_params_per_pair = sum(
            p.numel() for p in self.gp._get_model(0, 0).parameters())
        gp_pair_total = gp_params_per_pair * self.N * self.N

        return {
            "vgae_params": vgae_params,
            "vgae_trainable": vgae_trainable,
            "feat_extractor_params": feat_params,
            "gp_params_per_pair": gp_params_per_pair,
            "gp_pair_total": gp_pair_total,
            "gp_total": feat_params + gp_pair_total,
            "total": vgae_params + feat_params + gp_pair_total,
        }

    def estimate_gflops(self) -> Dict[str, float]:
        N = self.N
        P = self.P
        d = self.vgae.d
        H = self.vgae.enc_gcn1.lin.out_features

        enc_in = P + N + 1
        dec_in = P + d + 2

        macs = 0
        # --- Encoder ---
        macs += N * N * enc_in + N * enc_in * H
        macs += N * N * H + N * H * d
        macs += N * N * H + N * H * d
        # --- Decoder ---
        macs += N * N * dec_in + N * dec_in * H
        macs += N * N * H + N * H * 1

        # --- Feature extractor (per source variable) ---
        feat_ext = self.gp.feature_extractor
        layers = [m for m in feat_ext.net if isinstance(m, nn.Linear)]
        feat_macs = 0
        for lin in layers:
            feat_macs += lin.in_features * lin.out_features
        # Applied N times (once per source variable)
        macs += N * feat_macs

        gflops = macs / 1e9
        mflops = macs / 1e6
        return {"macs": macs, "gflops": gflops, "mflops": mflops}

    @torch.no_grad()
    def measure_latency(
        self,
        x_ab_list,
        n_warmup: int = 3,
        n_runs: int | None = None,
    ) -> Dict[str, float]:
        self.vgae.eval()
        windows = x_ab_list
        if n_runs is not None:
            windows = windows[:n_runs]

        for i in range(min(n_warmup, len(windows))):
            Xw = windows[i]
            if Xw.ndim == 3:
                Xw = Xw.squeeze(0) if Xw.shape[0] == 1 else Xw.reshape(-1, Xw.shape[-1])
            self.rca_on_window(Xw)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for Xw in windows:
            if Xw.ndim == 3:
                Xw = Xw.squeeze(0) if Xw.shape[0] == 1 else Xw.reshape(-1, Xw.shape[-1])
            self.rca_on_window(Xw)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        n = len(windows)
        return {
            "total_sec": elapsed,
            "per_window_ms": (elapsed / n * 1000) if n > 0 else 0.0,
            "n_windows": n,
        }
