"""DyCauseNet v3: Bypass-blocked dynamic causal network.

Key design rules (bypass blocking):
    1. μ̂ prediction ONLY through A (causal coefficients) — no context head
    2. σ̂² from target variable's OWN lag only — no cross-variable input
    3. Cross-variable interaction ONLY through A_t

Architecture:
    - Lag-specific causal coefficients A^(k)_{j,i,t} with OU prior
    - Factorized inference net with temporal bottleneck → q(A_t)
    - Fixed basis: b(y) = y + MLP(y)  (identity skip + nonlinear residual)
    - μ̂_j = Σ_{i,k} A^(k) · b(y_{i,t-k+1}) + bias_j
    - σ̂²_j = softplus(f(y_{j,t:t-L+1})) + ε
    - Edge-wise group sparsity (time-pooled L2/L1 for edge selection)
    - Exogenous independence: off-diagonal covariance penalty on residuals

Loss (ELBO + structural):
    L = NLL + β·KL + λ·edge_group_sparsity + η·exog_independence
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from tqdm import tqdm

from .basis import LagBasis
from .encoder import InferenceNet
from .ou_dynamics import OUPrior
from .additive_model import build_hist_and_target, CausalMeanOnly, VarHead


class DyCauseNet(nn.Module):
    """Bypass-blocked dynamic causal network.

    Args:
        num_vars: N
        lag: L (lag window)
        basis_hidden: hidden dim for basis MLP residual
        gru_hidden: InferenceNet GRU hidden dim
        bottleneck_dim: InferenceNet temporal bottleneck dim
        emb_dim: embedding dim for (j, i, k) in InferenceNet
        mlp_hidden: InferenceNet MLP hidden dim
        var_hidden: VarHead hidden dim
        rho_init: OU AR coefficient init
        q_init: OU process noise init
        kl_max: maximum β for KL annealing
        sparse_weight: λ for edge-wise group sparsity
        exog_weight: η for exogenous independence
        device: torch device
    """

    def __init__(
        self,
        num_vars: int,
        lag: int = 5,
        basis_hidden: int = 32,
        gru_hidden: int = 128,
        bottleneck_dim: int = 8,
        emb_dim: int = 16,
        mlp_hidden: int = 64,
        var_hidden: int = 32,
        rho_init: float = 0.95,
        q_init: float = 0.01,
        kl_max: float = 1.0,
        sparse_weight: float = 1e-3,
        exog_weight: float = 0.1,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.num_vars = num_vars
        self.lag = lag
        self.kl_max = kl_max
        self.sparse_weight = sparse_weight
        self.exog_weight = exog_weight
        self.device = device

        # Sub-modules
        self.basis = LagBasis(hidden=basis_hidden)
        self.inference_net = InferenceNet(
            num_vars, lag, gru_hidden=gru_hidden,
            bottleneck_dim=bottleneck_dim,
            emb_dim=emb_dim, mlp_hidden=mlp_hidden,
        )
        self.ou_prior = OUPrior(rho_init=rho_init, q_init=q_init)
        self.causal_mean = CausalMeanOnly(num_vars)
        self.var_head = VarHead(num_vars, lag, hidden=var_hidden)

        # Normal-phase baselines
        self.register_buffer("A_ref_mean", torch.zeros(num_vars, num_vars))
        self.register_buffer("A_ref_var", torch.ones(num_vars, num_vars))
        self.register_buffer("forecast_threshold", torch.tensor(0.0))

        self.fitted = False
        self.to(device)

    # ------------------------------------------------------------------
    #  Causal strength: aggregate A across lags
    # ------------------------------------------------------------------
    @staticmethod
    def agg_edge_strength(A_mu: torch.Tensor) -> torch.Tensor:
        """Aggregate causal strength across lags: Ā_{j,i} = sqrt(Σ_k A^(k)²).

        Args:
            A_mu: [..., N, N, L]

        Returns:
            Abar: [..., N, N]
        """
        return torch.sqrt((A_mu ** 2).sum(dim=-1) + 1e-8)

    # ------------------------------------------------------------------
    #  Core forward: ELBO + structural losses
    # ------------------------------------------------------------------
    def forward_sequence(
        self,
        Y: torch.Tensor,
        kl_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Process sequence and compute loss.

        L = NLL + β·KL + λ·sparsity + η·exog_independence

        Args:
            Y: [T, N] time series
            kl_weight: current KL weight (annealed)
        """
        T, N = Y.shape
        L = self.lag
        if T <= L:
            raise ValueError(f"Sequence length {T} must be > lag {L}")

        # Data alignment (leakage-free)
        X_hist, Y_next = build_hist_and_target(Y, L)  # [m, N, L], [m, N]
        m = X_hist.shape[0]

        # Inference: q(A_t)
        A_mu, A_logvar = self.inference_net(X_hist)  # [m, N, N, L]

        # Sample A (reparameterization)
        if self.training:
            eps = torch.randn_like(A_mu)
            A_sample = A_mu + torch.exp(0.5 * A_logvar) * eps
        else:
            A_sample = A_mu

        # Basis transform
        B_feat = self.basis(X_hist)  # [m, N, L]

        # Mean prediction (ONLY through A — no bypass)
        mu = self.causal_mean(A_sample, B_feat)  # [m, N]

        # Variance prediction (own-lag ONLY)
        sigma2 = self.var_head(X_hist)  # [m, N]

        # ── Loss 1: NLL ──
        nll = 0.5 * (torch.log(sigma2) + (Y_next - mu) ** 2 / sigma2)
        loss_nll = nll.mean()

        # ── Loss 2: KL (OU prior) ──
        loss_kl = self.ou_prior.kl(A_mu, A_logvar)

        # ── Loss 3: Edge-wise group sparsity (time-pooled) ──
        # For each edge (j,i): sqrt(mean_t[Σ_k A^(k)²] + ε)
        # Encourages *edge selection* (few edges, large A) over uniform shrinkage
        Abar = self.agg_edge_strength(A_mu)  # [m, N, N]
        edge_group = torch.sqrt(Abar.pow(2).mean(dim=0) + 1e-12)  # [N, N]
        loss_sparse = edge_group.mean()

        # ── Loss 4: Exogenous independence (off-diagonal covariance penalty) ──
        loss_exog = torch.tensor(0.0, device=self.device)
        if self.exog_weight > 0 and m > 1:
            residual = Y_next - mu  # [m, N]
            U = residual - residual.mean(dim=0, keepdim=True)
            Cov = (U.t() @ U) / (m - 1 + 1e-8)  # [N, N]
            off_diag = Cov - torch.diag(torch.diag(Cov))
            loss_exog = off_diag.pow(2).mean()

        # ── Total loss ──
        loss = (
            loss_nll
            + kl_weight * loss_kl
            + self.sparse_weight * loss_sparse
            + self.exog_weight * loss_exog
        )

        with torch.no_grad():
            Abar_det = Abar.detach()

        return {
            "loss": loss,
            "loss_nll": loss_nll,
            "loss_kl": loss_kl,
            "loss_sparse": loss_sparse,
            "loss_exog": loss_exog,
            "A_mu": A_mu.detach(),         # [m, N, N, L]
            "A_logvar": A_logvar.detach(),  # [m, N, N, L]
            "Abar": Abar_det,              # [m, N, N]
            "mu": mu.detach(),             # [m, N]
            "sigma2": sigma2.detach(),     # [m, N]
        }

    # ------------------------------------------------------------------
    #  Training
    # ------------------------------------------------------------------
    def fit(
        self,
        normal_sequences: List[torch.Tensor],
        epochs: int = 300,
        lr: float = 1e-3,
        kl_anneal_epochs: int = 15,
        max_seq_len: int = 500,
        forecast_quantile: float = 0.99,
        save_dir: Optional[str] = None,
        verbose: bool = True,
    ):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        seq_tensors = []
        for s in normal_sequences:
            seq = self._to_tensor(s).to(self.device)
            if seq.shape[0] > self.lag + 1:
                seq_tensors.append(seq)

        windows = []
        for seq_idx, seq in enumerate(seq_tensors):
            T_seq = seq.shape[0]
            for start in range(0, T_seq, max_seq_len):
                end = min(start + max_seq_len, T_seq)
                if end - start > self.lag + 1:
                    windows.append((seq_idx, start, end))
        n_windows = len(windows)
        if n_windows == 0:
            raise ValueError("No valid windows.")

        best_loss = float("inf")

        for epoch in range(epochs):
            kl_weight = self.kl_max * min(1.0, epoch / max(1, kl_anneal_epochs))

            order = torch.randperm(n_windows)
            epoch_loss = 0.0
            epoch_nll = 0.0
            epoch_kl = 0.0
            epoch_exog = 0.0

            win_pbar = range(n_windows)
            if verbose:
                win_pbar = tqdm(win_pbar, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

            for i in win_pbar:
                seq_idx, start, end = windows[order[i]]
                Y_seq = seq_tensors[seq_idx][start:end]

                optimizer.zero_grad()
                result = self.forward_sequence(Y_seq, kl_weight=kl_weight)
                result["loss"].backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += result["loss"].item()
                epoch_nll += result["loss_nll"].item()
                epoch_kl += result["loss_kl"].item()
                epoch_exog += result["loss_exog"].item()

                if verbose and isinstance(win_pbar, tqdm):
                    win_pbar.set_postfix({
                        "loss": f"{epoch_loss/(i+1):.4f}",
                        "nll": f"{epoch_nll/(i+1):.4f}",
                    })

            avg_loss = epoch_loss / n_windows
            avg_nll = epoch_nll / n_windows
            avg_kl = epoch_kl / n_windows
            avg_exog = epoch_exog / n_windows

            if verbose:
                star = " *" if avg_loss < best_loss else ""
                print(
                    f"[Epoch {epoch+1}/{epochs}] "
                    f"loss={avg_loss:.4f}  nll={avg_nll:.4f}  "
                    f"kl={avg_kl:.4f}  exog={avg_exog:.6f}  "
                    f"kl_w={kl_weight:.2f}  rho={self.ou_prior.rho.item():.3f}"
                    f"{star}"
                )

            if avg_loss < best_loss:
                best_loss = avg_loss
                self._best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                if save_dir is not None:
                    import os
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(self._best_state, os.path.join(save_dir, "model_best.pt"))

        if hasattr(self, "_best_state"):
            self.load_state_dict(self._best_state)
            if verbose:
                print(f"Restored best model (loss={best_loss:.4f})")

        self._compute_baselines(normal_sequences, max_seq_len, forecast_quantile)
        self.fitted = True

    @staticmethod
    def _to_tensor(arr) -> torch.Tensor:
        if isinstance(arr, np.ndarray):
            return torch.tensor(arr, dtype=torch.float32)
        return arr.float()

    def _compute_baselines(
        self,
        normal_sequences: List[torch.Tensor],
        max_seq_len: int = 500,
        forecast_quantile: float = 0.99,
    ):
        self.eval()
        all_Abar = []
        all_nll = []

        with torch.no_grad():
            for seq in normal_sequences:
                seq = self._to_tensor(seq).to(self.device)[:max_seq_len]
                if seq.shape[0] <= self.lag + 1:
                    continue
                result = self.forward_sequence(seq, kl_weight=1.0)
                all_Abar.append(result["Abar"])  # [m, N, N]

                # Per-step NLL for threshold
                nll_per_step = 0.5 * (
                    torch.log(result["sigma2"])
                    + (build_hist_and_target(seq, self.lag)[1] - result["mu"]) ** 2
                    / result["sigma2"]
                ).sum(dim=-1)  # [m]
                all_nll.append(nll_per_step)

        if all_Abar:
            A_cat = torch.cat(all_Abar, dim=0)
            self.A_ref_mean.copy_(A_cat.mean(dim=0))
            self.A_ref_var.copy_(A_cat.var(dim=0) + 1e-6)

        if all_nll:
            nll_cat = torch.cat(all_nll)
            q = float(np.clip(forecast_quantile, 0.5, 0.9999))
            self.forecast_threshold.copy_(torch.quantile(nll_cat, q))

    # ------------------------------------------------------------------
    #  Inference for a single window
    # ------------------------------------------------------------------
    @torch.no_grad()
    def infer_window(
        self,
        X_win: torch.Tensor,
        avg_last_k: int = 5,
    ) -> Dict[str, Any]:
        """Run inference on a window for RCA/detection."""
        self.eval()
        X_win = self._to_tensor(X_win).to(self.device)
        T, N = X_win.shape

        if T <= self.lag:
            return {
                "rca_scores": torch.zeros(N, device=self.device),
                "forecast_scores": torch.zeros(0, device=self.device),
                "forecast_anomaly": torch.zeros(0, dtype=torch.bool, device=self.device),
            }

        X_hist, Y_next = build_hist_and_target(X_win, self.lag)
        m = X_hist.shape[0]

        A_mu, A_logvar = self.inference_net(X_hist)
        B_feat = self.basis(X_hist)
        mu = self.causal_mean(A_mu, B_feat)
        sigma2 = self.var_head(X_hist)

        # Per-step forecast NLL
        nll_per_step = 0.5 * (
            torch.log(sigma2) + (Y_next - mu) ** 2 / sigma2
        ).sum(dim=-1)  # [m]
        forecast_anomaly = nll_per_step > self.forecast_threshold

        # RCA (last k steps)
        k = min(avg_last_k, m)

        # Standardized residual
        r = (Y_next[-k:] - mu[-k:]).abs() / (sigma2[-k:].sqrt() + 1e-8)  # [k, N]
        r_avg = r.mean(dim=0)  # [N]

        # Aggregate causal strength
        Abar = self.agg_edge_strength(A_mu[-k:])  # [k, N, N]
        Abar_avg = Abar.mean(dim=0)  # [N, N]

        # Causal deviation from normal baseline
        D = (Abar_avg - self.A_ref_mean).abs()  # [N, N]

        # Edge RCA: r̃_j · Ā_{j,i} + β · D_{j,i}
        edge_rca = r_avg.unsqueeze(1) * Abar_avg + D  # [N, N] (j, i)

        # Per-source score: RC(i) = Σ_j edge_rca[j, i]
        rca_scores = edge_rca.sum(dim=0)  # [N]

        return {
            "rca_scores": rca_scores,
            "forecast_scores": nll_per_step,
            "forecast_anomaly": forecast_anomaly,
            "A_mu": A_mu,
            "Abar_avg": Abar_avg,
            "residual": (Y_next[-k:] - mu[-k:]).mean(dim=0),
            "pred_std": sigma2[-k:].sqrt().mean(dim=0),
            "n_skipped": self.lag,
        }

    # ------------------------------------------------------------------
    #  Evaluation: RCA (AC@k)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate_rca(
        self,
        abnormal_windows: List[torch.Tensor],
        labels: List,
        k_list: List[int] = [1, 3, 5],
        avg_last_k: int = 5,
        **kwargs,
    ) -> Dict[str, float]:
        self.eval()
        hit_lists = {k: [] for k in k_list}
        max_k = max(k_list)

        for Xw, true_vars in zip(abnormal_windows, labels):
            if isinstance(true_vars, np.ndarray):
                if true_vars.size == 0:
                    continue
                if true_vars.ndim == 2:
                    true_vars = np.where(true_vars.sum(axis=0) > 0)[0].tolist()
                elif true_vars.ndim == 1:
                    true_vars = np.where(true_vars > 0)[0].tolist()
                else:
                    true_vars = true_vars.tolist()
            if not true_vars:
                continue

            result = self.infer_window(
                self._to_tensor(Xw).to(self.device),
                avg_last_k=avg_last_k,
            )
            scores = result["rca_scores"]
            topk = torch.topk(scores, k=min(max_k, scores.numel())).indices.tolist()

            for k in k_list:
                kk = min(k, len(topk))
                hit = 1.0 if any(v in topk[:kk] for v in true_vars) else 0.0
                hit_lists[k].append(hit)

        metrics = {}
        for k in k_list:
            metrics[f"ac@{k}"] = float(np.mean(hit_lists[k])) if hit_lists[k] else 0.0
        metrics["n"] = len(hit_lists[k_list[0]]) if k_list else 0
        return metrics

    # ------------------------------------------------------------------
    #  Evaluation: Forecasting anomaly detection
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate_detection(
        self,
        abnormal_windows: List[torch.Tensor],
        labels: List[np.ndarray],
        **kwargs,
    ) -> Dict[str, float]:
        self.eval()
        all_scores, all_targets = [], []

        for Xw, y in zip(abnormal_windows, labels):
            Xw = self._to_tensor(Xw).to(self.device)
            result = self.infer_window(Xw, avg_last_k=1)

            scores = result["forecast_scores"].cpu().numpy()
            if scores.size == 0:
                continue

            y_arr = np.asarray(y.cpu().numpy() if isinstance(y, torch.Tensor) else y)
            if y_arr.ndim == 2:
                y_any = (y_arr.sum(axis=1) > 0).astype(float)
            elif y_arr.ndim == 1:
                y_any = np.full(Xw.shape[0], float(y_arr.sum() > 0))
            else:
                continue

            n_skipped = int(result.get("n_skipped", self.lag))
            y_valid = y_any[n_skipped:n_skipped + scores.size]
            m = min(y_valid.size, scores.size)
            y_valid, scores = y_valid[:m], scores[:m]
            if m == 0:
                continue

            all_scores.append(scores)
            all_targets.append(y_valid)

        if not all_scores:
            return {"auroc": 0.0, "auprc": 0.0, "f1": 0.0, "n": 0}

        s = np.concatenate(all_scores)
        y = np.concatenate(all_targets).astype(float)
        y_hat = (s >= float(self.forecast_threshold.item())).astype(float)

        out = {"n": int(s.size), "threshold": float(self.forecast_threshold.item())}
        try:
            out["auroc"] = float(roc_auc_score(y, s))
        except ValueError:
            out["auroc"] = 0.0
        try:
            out["auprc"] = float(average_precision_score(y, s))
        except ValueError:
            out["auprc"] = 0.0

        tp = float(((y_hat == 1) & (y == 1)).sum())
        fp = float(((y_hat == 1) & (y == 0)).sum())
        fn = float(((y_hat == 0) & (y == 1)).sum())
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        out["precision"] = prec
        out["recall"] = rec
        out["f1"] = 2.0 * prec * rec / (prec + rec + 1e-8)
        return out

    # ------------------------------------------------------------------
    #  Evaluation: Causal Discovery
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate_causal_discovery(
        self,
        causal_struct: np.ndarray,
        normal_sequences: List[torch.Tensor],
        max_seq_len: int = 500,
    ) -> Dict[str, float]:
        """Evaluate causal discovery using time-averaged Ā = Σ_k |A^(k)_mu|."""
        self.eval()
        N = self.num_vars

        all_Abar = []
        for seq in normal_sequences:
            seq = self._to_tensor(seq).to(self.device)[:max_seq_len]
            if seq.shape[0] <= self.lag + 1:
                continue
            result = self.forward_sequence(seq, kl_weight=1.0)
            all_Abar.append(result["Abar"].mean(dim=0))  # [N, N]

        if not all_Abar:
            return {"auroc": 0.0, "auprc": 0.0, "f1": 0.0}

        avg_Abar = torch.stack(all_Abar).mean(dim=0).cpu().numpy()
        gt = causal_struct.astype(float).ravel()
        scores = avg_Abar.ravel()

        results = {}
        try:
            results["auroc"] = float(roc_auc_score(gt, scores))
        except ValueError:
            results["auroc"] = 0.0
        try:
            results["auprc"] = float(average_precision_score(gt, scores))
        except ValueError:
            results["auprc"] = 0.0

        threshold = np.percentile(scores, 70)
        pred_binary = (scores >= threshold).astype(float)
        results["f1"] = float(f1_score(gt, pred_binary, zero_division=0))
        results["score_method"] = "avg_Abar"

        return results

    # ------------------------------------------------------------------
    #  Causal summary
    # ------------------------------------------------------------------
    @torch.no_grad()
    def causal_summary(
        self,
        normal_sequences: List[torch.Tensor],
        max_seq_len: int = 500,
    ) -> Dict[str, Any]:
        self.eval()
        all_Abar = []

        for seq in normal_sequences:
            seq = self._to_tensor(seq).to(self.device)[:max_seq_len]
            if seq.shape[0] <= self.lag + 1:
                continue
            result = self.forward_sequence(seq, kl_weight=1.0)
            all_Abar.append(result["Abar"])

        N = self.num_vars
        if not all_Abar:
            return {
                "A_mean": torch.zeros(N, N),
                "A_var": torch.zeros(N, N),
                "ou_rho": self.ou_prior.rho.item(),
                "ou_Q": self.ou_prior.Q.item(),
                "forecast_threshold": float(self.forecast_threshold.item()),
            }

        A_cat = torch.cat(all_Abar, dim=0)
        return {
            "A_mean": A_cat.mean(dim=0).cpu(),
            "A_var": A_cat.var(dim=0).cpu(),
            "ou_rho": self.ou_prior.rho.item(),
            "ou_Q": self.ou_prior.Q.item(),
            "forecast_threshold": float(self.forecast_threshold.item()),
        }

    # ------------------------------------------------------------------
    #  Model info
    # ------------------------------------------------------------------
    def count_parameters(self) -> Dict[str, int]:
        def _count(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        return {
            "basis": _count(self.basis),
            "inference_net": _count(self.inference_net),
            "ou_prior": _count(self.ou_prior),
            "causal_mean": _count(self.causal_mean),
            "var_head": _count(self.var_head),
            "total": _count(self),
        }
