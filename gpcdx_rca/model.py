"""DyCauseNet: Dynamic Causal Network with OU-based time-varying inference.

End-to-end model that jointly learns:
    1. Time-varying causal weights W_t via OU dynamics + amortized VI
    2. Additive Granger prediction f(x; W_t)
    3. Low-rank + diagonal residual model for exogenous scoring
    4. RCA scoring combining causal change + exogenous anomaly
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from tqdm import tqdm

from .ou_dynamics import OUDynamics
from .additive_model import NeuralGrangerModel
from .encoder import CausalStateEncoder
from .residual_model import LowRankResidualModel


class DyCauseNet(nn.Module):
    """Dynamic Causal Network for time-varying Granger causality + RCA.

    Architecture:
        - OU dynamics: Z_t evolves as AR(1) with mean-reversion
        - Additive prediction: x̂_i = Σ_j W_{ij,t} · ψ̃_j(u_j)
        - Amortized encoder: q(Z_t | Z_{t-1}, x_{≤t})
        - Residual model: r_t ~ N(0, BB⊤ + D)

    Training (ELBO):
        max  E_q[log p(x|W)] - KL(q(Z)||p_OU(Z)) - λ_s|W|_1
    """

    def __init__(
        self,
        num_vars: int,
        lag: int = 5,
        hidden_dim: int = 64,
        feat_dim: int = 8,
        encoder_hidden: int = 128,
        rank: int = 2,
        rho_init: float = 0.95,
        q_init: float = 0.01,
        mu_z_init: float = 0.0,
        sparse_weight: float = 0.01,
        cancel_weight: float = 0.01,
        kl_max: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.num_vars = num_vars
        self.lag = lag
        self.sparse_weight = sparse_weight
        self.cancel_weight = cancel_weight
        self.kl_max = kl_max
        self.device = device

        # Sub-modules: signed decomposition (positive/negative channels)
        self.ou_pos = OUDynamics(num_vars, rho_init=rho_init, q_init=q_init, mu_z_init=mu_z_init)
        self.ou_neg = OUDynamics(num_vars, rho_init=rho_init, q_init=q_init, mu_z_init=mu_z_init)
        self.additive = NeuralGrangerModel(num_vars, lag, feat_dim=feat_dim, hidden_dim=hidden_dim)
        self.encoder_pos = CausalStateEncoder(num_vars, encoder_hidden)
        self.encoder_neg = CausalStateEncoder(num_vars, encoder_hidden)
        self.residual = LowRankResidualModel(num_vars, rank)

        # Normal-phase baselines (populated during training)
        self.register_buffer("W_baseline_mean", torch.zeros(num_vars, num_vars))
        self.register_buffer("W_baseline_var", torch.ones(num_vars, num_vars))
        self.register_buffer("forecast_threshold", torch.tensor(0.0))
        self.register_buffer("forecast_score_mean", torch.tensor(0.0))
        self.register_buffer("forecast_score_std", torch.tensor(1.0))

        self.fitted = False
        self.to(device)

    # ------------------------------------------------------------------
    #  Core forward: encode a sequence and compute ELBO
    # ------------------------------------------------------------------
    def _signed_transform(
        self,
        z_pos: torch.Tensor,
        z_neg: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map signed latent channels to (W, W_pos, W_neg)."""
        w_pos = self.ou_pos.z_to_positive(z_pos)
        w_neg = self.ou_neg.z_to_positive(z_neg)
        return w_pos - w_neg, w_pos, w_neg

    def forward_sequence(
        self,
        X_seq: torch.Tensor,
        kl_weight: float = 1.0,
        mse_weight: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """Process a sequence with forecasting NLL + MSE regularization.

        Loss = NLL(r) + mse_weight*MSE + kl_weight*KL + sparse*|W| + cancel*(W⁺·W⁻)

        NLL is the primary driver (proper likelihood for structure learning).
        MSE acts as auxiliary regularizer to stabilize early training.
        """
        T, N = X_seq.shape
        lag = self.lag
        if T <= lag:
            raise ValueError(f"Sequence length {T} must be > lag {lag}")

        lag_windows, targets = self.additive.build_lag_windows(X_seq)
        m = lag_windows.shape[0]

        obs_seq = X_seq[lag:]
        z_init_pos = self.ou_pos.mu_Z.detach()
        z_init_neg = self.ou_neg.mu_Z.detach()
        q_mu_pos, q_logvar_pos = self.encoder_pos.encode_sequence(obs_seq, z_init_pos)
        q_mu_neg, q_logvar_neg = self.encoder_neg.encode_sequence(obs_seq, z_init_neg)

        if self.training:
            eps_pos = torch.randn_like(q_mu_pos)
            eps_neg = torch.randn_like(q_mu_neg)
            z_s_pos = q_mu_pos + torch.exp(0.5 * q_logvar_pos) * eps_pos
            z_s_neg = q_mu_neg + torch.exp(0.5 * q_logvar_neg) * eps_neg
        else:
            z_s_pos = q_mu_pos
            z_s_neg = q_mu_neg

        w_samples, w_pos_samples, w_neg_samples = self._signed_transform(z_s_pos, z_s_neg)
        w_point, w_pos_point, w_neg_point = self._signed_transform(q_mu_pos, q_mu_neg)

        x_hat = self.additive(lag_windows, w_samples)
        residuals = targets - x_hat

        # PRIMARY: Forecasting NLL — proper likelihood drives structure learning
        nll_per_step = self.residual.nll_per_sample(residuals)
        nll_per_step = nll_per_step.clamp(max=50.0 * N)  # prevent gradient explosion
        pred_nll = nll_per_step.mean()

        # AUXILIARY: MSE regularizer — stabilizes early training
        mse_loss = (residuals ** 2).mean()

        total_kl = torch.tensor(0.0, device=self.device)

        # t=0
        total_kl = total_kl + self.ou_pos.kl_t0(q_mu_pos[0], q_logvar_pos[0])
        total_kl = total_kl + self.ou_neg.kl_t0(q_mu_neg[0], q_logvar_neg[0])

        allow_temporal = getattr(self, '_allow_temporal_grad', False)
        for t in range(1, m):
            z_prev_pos = q_mu_pos[t - 1] if allow_temporal else q_mu_pos[t - 1].detach()
            z_prev_neg = q_mu_neg[t - 1] if allow_temporal else q_mu_neg[t - 1].detach()
            total_kl = total_kl + self.ou_pos.kl_transition(
                q_mu_pos[t], q_logvar_pos[t], z_prev_pos
            )
            total_kl = total_kl + self.ou_neg.kl_transition(
                q_mu_neg[t], q_logvar_neg[t], z_prev_neg
            )

        kl_loss = total_kl / (m * N)

        # Signed-structure regularization: sparsity + cancellation penalty
        sparse_loss = w_pos_samples.abs().mean() + w_neg_samples.abs().mean()
        cancel_loss = (w_pos_samples * w_neg_samples).mean()

        loss = (
            pred_nll
            + mse_weight * mse_loss
            + kl_weight * kl_loss
            + self.sparse_weight * sparse_loss
            + self.cancel_weight * cancel_loss
        )

        return {
            "loss": loss,
            "pred_nll": pred_nll,
            "mse_loss": mse_loss,
            "kl_loss": kl_loss,
            "sparse_loss": sparse_loss,
            "cancel_loss": cancel_loss,
            "W_mean_seq": w_point.detach(),  # transformed posterior mean path
            "W_pos_seq": w_pos_point.detach(),
            "W_neg_seq": w_neg_point.detach(),
            "Z_mu_pos": q_mu_pos.detach(),
            "Z_mu_neg": q_mu_neg.detach(),
            "Z_var_pos": torch.exp(q_logvar_pos).detach(),
            "Z_var_neg": torch.exp(q_logvar_neg).detach(),
            "residual_seq": residuals.detach(),
            "pred_nll_seq": nll_per_step.detach(),
        }

    # ------------------------------------------------------------------
    #  Training
    # ------------------------------------------------------------------
    def fit(
        self,
        normal_sequences: List[torch.Tensor],
        epochs: int = 30,
        lr: float = 1e-3,
        kl_anneal_epochs: int = 10,
        max_seq_len: int = 500,
        forecast_quantile: float = 0.99,
        save_dir: Optional[str] = None,
        verbose: bool = True,
    ):
        """Train on normal data using ELBO.

        Args:
            normal_sequences: list of (T_i, N) tensors
            epochs: number of training epochs
            lr: learning rate
            kl_anneal_epochs: epochs over which KL weight ramps 0→1
            max_seq_len: maximum sequence length per iteration
            forecast_quantile: train quantile for predictive NLL threshold
            verbose: show progress bar
        """
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Convert numpy arrays to tensors and keep sequence boundaries.
        seq_tensors = []
        for s in normal_sequences:
            if isinstance(s, np.ndarray):
                seq = torch.tensor(s, dtype=torch.float32)
            else:
                seq = s.float()
            if seq.shape[0] > self.lag + 1:
                seq_tensors.append(seq.to(self.device))

        # Build non-overlapping windows *within each sequence*.
        # This avoids artificial transitions across sequence boundaries.
        windows = []
        for seq_idx, seq in enumerate(seq_tensors):
            T_seq = seq.shape[0]
            for start in range(0, T_seq, max_seq_len):
                end = min(start + max_seq_len, T_seq)
                if end - start > self.lag + 1:
                    windows.append((seq_idx, start, end))
        n_windows = len(windows)
        if n_windows == 0:
            raise ValueError("No valid windows found. Increase sequence length or reduce lag.")

        best_loss = float("inf")

        for epoch in range(epochs):
            # KL annealing: linearly ramp from 0 to kl_max
            kl_weight = self.kl_max * min(1.0, epoch / max(1, kl_anneal_epochs))

            # Tau annealing: warm (smooth) → cold (sharp) sigmoid gate
            progress = epoch / max(1, epochs - 1)
            self.ou_pos.set_tau(progress)
            self.ou_neg.set_tau(progress)

            # Temporal gradient: detach z_prev only during warmup (A2 fix)
            self._allow_temporal_grad = (epoch >= kl_anneal_epochs)

            # Shuffle window order each epoch
            order = torch.randperm(n_windows)
            epoch_loss = 0.0
            epoch_nll = 0.0
            epoch_mse = 0.0
            epoch_cancel = 0.0

            win_pbar = range(n_windows)
            if verbose:
                win_pbar = tqdm(
                    win_pbar,
                    desc=f"Epoch {epoch+1}/{epochs}",
                    leave=False,
                )

            for i in win_pbar:
                seq_idx, start, end = windows[order[i]]
                X_seq = seq_tensors[seq_idx][start:end]

                optimizer.zero_grad()
                result = self.forward_sequence(X_seq, kl_weight=kl_weight)
                result["loss"].backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += result["loss"].item()
                epoch_nll += result["pred_nll"].item()
                epoch_mse += result["mse_loss"].item()
                epoch_cancel += result["cancel_loss"].item()

                if verbose and isinstance(win_pbar, tqdm):
                    win_pbar.set_postfix({
                        "loss": f"{epoch_loss / (i+1):.4f}",
                        "nll": f"{epoch_nll / (i+1):.4f}",
                    })

            avg_loss = epoch_loss / n_windows
            avg_nll = epoch_nll / n_windows
            avg_mse = epoch_mse / n_windows
            avg_cancel = epoch_cancel / n_windows

            if verbose:
                star = ""
                if avg_loss < best_loss:
                    star = " *"
                print(
                    f"[Epoch {epoch+1}/{epochs}] "
                    f"loss={avg_loss:.4f}  nll={avg_nll:.4f}  "
                    f"mse={avg_mse:.4f}  cancel={avg_cancel:.4f}  "
                    f"kl_w={kl_weight:.2f}  tau={self.ou_pos.tau.item():.2f}  "
                    f"rho_mean={(0.5 * (self.ou_pos.rho.mean() + self.ou_neg.rho.mean())).item():.3f}"
                    f"{star}"
                )

            # Save best model checkpoint
            if avg_loss < best_loss:
                best_loss = avg_loss
                self._best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                if save_dir is not None:
                    import os
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(self._best_state, os.path.join(save_dir, "model_best.pt"))

        # Restore best weights
        if hasattr(self, "_best_state"):
            self.load_state_dict(self._best_state)
            if verbose:
                print(f"Restored best model (loss={best_loss:.4f})")

        # Set tau to final (cold) value for evaluation
        self.ou_pos.set_tau(1.0)
        self.ou_neg.set_tau(1.0)

        # --- Compute normal baselines ---
        self._compute_baselines(
            normal_sequences,
            max_seq_len=max_seq_len,
            forecast_quantile=forecast_quantile,
        )
        self.fitted = True

    @staticmethod
    def _to_tensor(arr) -> torch.Tensor:
        """Convert numpy array to float tensor if needed."""
        if isinstance(arr, np.ndarray):
            return torch.tensor(arr, dtype=torch.float32)
        return arr.float()

    def _compute_baselines(
        self,
        normal_sequences: List[torch.Tensor],
        max_seq_len: int = 500,
        forecast_quantile: float = 0.99,
    ):
        """Compute normal baselines for RCA and forecasting detection."""
        self.eval()
        all_w = []
        all_scores = []

        with torch.no_grad():
            for seq in normal_sequences:
                seq = self._to_tensor(seq).to(self.device)
                if seq.shape[0] <= self.lag + 1:
                    continue
                seq = seq[:max_seq_len]
                result = self.forward_sequence(seq, kl_weight=1.0)
                all_w.append(result["W_mean_seq"])
                all_scores.append(result["pred_nll_seq"])

        if all_w:
            all_w_cat = torch.cat(all_w, dim=0)  # (total_m, N, N)
            self.W_baseline_mean.copy_(all_w_cat.mean(dim=0))
            self.W_baseline_var.copy_(all_w_cat.var(dim=0) + 1e-6)

        if all_scores:
            score_cat = torch.cat(all_scores, dim=0)
            self.forecast_score_mean.copy_(score_cat.mean())
            self.forecast_score_std.copy_(score_cat.std().clamp_min(1e-6))
            q = float(np.clip(forecast_quantile, 0.5, 0.9999))
            threshold = torch.quantile(score_cat, q)
            self.forecast_threshold.copy_(threshold)

    # ------------------------------------------------------------------
    #  Online filtering state API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def init_filter_state(
        self,
        lag_history: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Initialize online filtering state.

        Args:
            lag_history: optional warm-start history (L, N). If L > lag,
                only the latest `lag` observations are kept.

        Returns:
            state dict containing GRU hidden, posterior Z summary, lag buffer.
        """
        if lag_history is None:
            lag_buffer = torch.zeros(0, self.num_vars, device=self.device)
        else:
            lag_buffer = self._to_tensor(lag_history).to(self.device)
            if lag_buffer.dim() != 2 or lag_buffer.shape[1] != self.num_vars:
                raise ValueError(
                    f"lag_history must have shape (L, {self.num_vars}), "
                    f"got {tuple(lag_buffer.shape)}"
                )
            if lag_buffer.shape[0] > self.lag:
                lag_buffer = lag_buffer[-self.lag:]

        z0_var_pos = self.ou_pos.stationary_variance.detach().clamp_min(1e-6)
        z0_var_neg = self.ou_neg.stationary_variance.detach().clamp_min(1e-6)
        return {
            "h_pos": self.encoder_pos.init_hidden(self.device),
            "h_neg": self.encoder_neg.init_hidden(self.device),
            "z_post_mu_pos": self.ou_pos.mu_Z.detach().clone(),
            "z_post_mu_neg": self.ou_neg.mu_Z.detach().clone(),
            "z_post_logvar_pos": torch.log(z0_var_pos),
            "z_post_logvar_neg": torch.log(z0_var_neg),
            "lag_buffer": lag_buffer,
            "num_obs": torch.tensor(float(lag_buffer.shape[0]), device=self.device),
        }

    @torch.no_grad()
    def _online_filter_step(
        self,
        x_t: torch.Tensor,
        state: Dict[str, torch.Tensor],
        n_mc_samples: int = 10,
    ) -> Dict[str, Any]:
        """Single online filtering update.

        Args:
            x_t: (N,) current observation
            state: online state from previous timestep
            n_mc_samples: number of posterior MC samples for W uncertainty

        Returns:
            dict with next state, posterior summaries, and (if ready) residual.
        """
        x_t = self._to_tensor(x_t).to(self.device).view(-1)
        if x_t.numel() != self.num_vars:
            raise ValueError(
                f"x_t must have {self.num_vars} elements, got {x_t.numel()}"
            )

        h_prev_pos = state["h_pos"]
        h_prev_neg = state["h_neg"]
        z_prev_mu_pos = state["z_post_mu_pos"]
        z_prev_mu_neg = state["z_post_mu_neg"]
        lag_buffer = state["lag_buffer"]

        # OU one-step prediction then amortized posterior update.
        z_prior_mu_pos, _ = self.ou_pos.transition(z_prev_mu_pos)
        z_prior_mu_neg, _ = self.ou_neg.transition(z_prev_mu_neg)

        q_mu_pos, q_logvar_pos, h_new_pos = self.encoder_pos.forward(
            x_t, h_prev_pos, z_prior_mu_pos
        )
        q_mu_neg, q_logvar_neg, h_new_neg = self.encoder_neg.forward(
            x_t, h_prev_neg, z_prior_mu_neg
        )
        q_var_pos = torch.exp(q_logvar_pos)
        q_var_neg = torch.exp(q_logvar_neg)

        n_mc = max(1, int(n_mc_samples))
        if n_mc == 1:
            w_draw = self._signed_transform(q_mu_pos, q_mu_neg)[0]
            W_draws = w_draw.unsqueeze(0)
        else:
            eps_pos = torch.randn(n_mc, self.num_vars, self.num_vars, device=self.device)
            eps_neg = torch.randn(n_mc, self.num_vars, self.num_vars, device=self.device)
            z_draws_pos = q_mu_pos.unsqueeze(0) + q_var_pos.sqrt().unsqueeze(0) * eps_pos
            z_draws_neg = q_mu_neg.unsqueeze(0) + q_var_neg.sqrt().unsqueeze(0) * eps_neg
            W_draws = self._signed_transform(z_draws_pos, z_draws_neg)[0]

        W_step_mean = W_draws.mean(dim=0)
        W_step_var = W_draws.var(dim=0, unbiased=False)

        ready = lag_buffer.shape[0] >= self.lag
        x_hat = None
        residual = None
        if ready:
            # lag_buffer stores chronological order. Model expects newest-first.
            lag_window = lag_buffer[-self.lag:].flip(0).transpose(0, 1).unsqueeze(0)
            x_hat = self.additive(lag_window, W_step_mean).squeeze(0)  # (N,)
            residual = x_t - x_hat

        next_lag = torch.cat([lag_buffer, x_t.unsqueeze(0)], dim=0)
        if next_lag.shape[0] > self.lag:
            next_lag = next_lag[-self.lag:]

        next_state = {
            "h_pos": h_new_pos,
            "h_neg": h_new_neg,
            "z_post_mu_pos": q_mu_pos,
            "z_post_mu_neg": q_mu_neg,
            "z_post_logvar_pos": q_logvar_pos,
            "z_post_logvar_neg": q_logvar_neg,
            "lag_buffer": next_lag,
            "num_obs": state["num_obs"] + 1.0,
        }

        return {
            "state": next_state,
            "ready": ready,
            "W_draws": W_draws,
            "W_step_mean": W_step_mean,
            "W_step_var": W_step_var,
            "x_hat": x_hat,
            "residual": residual,
        }

    @torch.no_grad()
    def _run_online_filter(
        self,
        X_seq: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
        n_mc_samples: int = 10,
    ) -> Dict[str, Any]:
        """Run online filtering over a sequence."""
        X_seq = self._to_tensor(X_seq).to(self.device)
        if X_seq.dim() != 2 or X_seq.shape[1] != self.num_vars:
            raise ValueError(
                f"X_seq must have shape (T, {self.num_vars}), got {tuple(X_seq.shape)}"
            )

        if state is None:
            state = self.init_filter_state()
        else:
            state = {
                "h_pos": state["h_pos"].to(self.device),
                "h_neg": state["h_neg"].to(self.device),
                "z_post_mu_pos": state["z_post_mu_pos"].to(self.device),
                "z_post_mu_neg": state["z_post_mu_neg"].to(self.device),
                "z_post_logvar_pos": state["z_post_logvar_pos"].to(self.device),
                "z_post_logvar_neg": state["z_post_logvar_neg"].to(self.device),
                "lag_buffer": state["lag_buffer"].to(self.device),
                "num_obs": state["num_obs"].to(self.device),
            }

        W_means, W_vars, W_draws, residuals, preds = [], [], [], [], []
        n_skipped = 0

        for t in range(X_seq.shape[0]):
            step = self._online_filter_step(
                X_seq[t], state, n_mc_samples=n_mc_samples
            )
            state = step["state"]
            if not step["ready"]:
                n_skipped += 1
                continue

            W_means.append(step["W_step_mean"])
            W_vars.append(step["W_step_var"])
            W_draws.append(step["W_draws"])
            residuals.append(step["residual"])
            preds.append(step["x_hat"])

        n_valid = len(W_means)
        N = self.num_vars
        n_mc = max(1, int(n_mc_samples))
        if n_valid == 0:
            return {
                "state": state,
                "n_valid": 0,
                "n_skipped": n_skipped,
                "W_means_seq": torch.zeros(0, N, N, device=self.device),
                "W_vars_seq": torch.zeros(0, N, N, device=self.device),
                "W_draws_seq": torch.zeros(0, n_mc, N, N, device=self.device),
                "residual_seq": torch.zeros(0, N, device=self.device),
                "pred_seq": torch.zeros(0, N, device=self.device),
            }

        return {
            "state": state,
            "n_valid": n_valid,
            "n_skipped": n_skipped,
            "W_means_seq": torch.stack(W_means, dim=0),      # (m, N, N)
            "W_vars_seq": torch.stack(W_vars, dim=0),        # (m, N, N)
            "W_draws_seq": torch.stack(W_draws, dim=0),      # (m, n_mc, N, N)
            "residual_seq": torch.stack(residuals, dim=0),   # (m, N)
            "pred_seq": torch.stack(preds, dim=0),           # (m, N)
        }

    # ------------------------------------------------------------------
    #  Online inference (single window)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def infer_window(
        self,
        X_win: torch.Tensor,
        avg_last_k: int = 5,
        n_mc_samples: int = 10,
        filter_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """Run online-filtering inference on a single window for RCA.

        Args:
            X_win: (T, N) observation window
            avg_last_k: average over last K timesteps for stability
            n_mc_samples: number of MC samples for Var(W|x)
            filter_state: optional previous online state for streaming

        Returns:
            dict with W_mean, W_var_posterior, residuals, rca_scores, etc.
        """
        self.eval()
        X_win = self._to_tensor(X_win).to(self.device)
        T, N = X_win.shape

        if T == 0:
            out = {
                "rca_scores": torch.zeros(N, device=self.device),
                "forecast_scores": torch.zeros(0, device=self.device),
                "forecast_anomaly": torch.zeros(0, dtype=torch.bool, device=self.device),
            }
            out["next_filter_state"] = (
                self.init_filter_state() if filter_state is None else filter_state
            )
            return out

        filtered = self._run_online_filter(
            X_win,
            state=filter_state,
            n_mc_samples=n_mc_samples,
        )

        if filtered["n_valid"] == 0:
            out = {
                "rca_scores": torch.zeros(N, device=self.device),
                "forecast_scores": torch.zeros(0, device=self.device),
                "forecast_anomaly": torch.zeros(0, dtype=torch.bool, device=self.device),
            }
            out["next_filter_state"] = filtered["state"]
            out["W_means_seq"] = filtered["W_means_seq"]
            return out

        k = min(avg_last_k, filtered["n_valid"])
        W_draws_last = filtered["W_draws_seq"][-k:].reshape(-1, N, N)
        W_mu = W_draws_last.mean(dim=0)                           # (N, N)
        W_var_posterior = W_draws_last.var(dim=0, unbiased=False) # (N, N)
        residual = filtered["residual_seq"][-k:].mean(dim=0)      # (N,)
        forecast_scores = self.residual.nll_per_sample(filtered["residual_seq"])  # (m,)
        forecast_anomaly = forecast_scores > self.forecast_threshold

        # RCA scores — now using posterior uncertainty
        rca_scores = self._compute_rca_scores(W_mu, residual, W_var_posterior)

        return {
            "W_mean": W_mu,
            "W_var_posterior": W_var_posterior,
            "residual": residual,
            "rca_scores": rca_scores,
            "W_means_seq": filtered["W_means_seq"],
            "forecast_scores": forecast_scores,
            "forecast_anomaly": forecast_anomaly,
            "forecast_score_last": forecast_scores[-1],
            "forecast_score_mean_last_k": forecast_scores[-k:].mean(),
            "n_skipped": filtered["n_skipped"],
            "next_filter_state": filtered["state"],
        }

    # ------------------------------------------------------------------
    #  RCA scoring
    # ------------------------------------------------------------------
    def _compute_rca_scores(
        self,
        W_mu: torch.Tensor,
        residual: torch.Tensor,
        W_var_posterior: Optional[torch.Tensor] = None,
        lambda_r: float = 1.0,
        lambda_w: float = 1.0,
        lambda_x: float = 0.5,
    ) -> torch.Tensor:
        """Compute per-node RCA scores.

        Combines three signals:
          1. Exogenous anomaly: per-node Mahalanobis score under residual model
          2. Causal change: deviation penalized by estimation uncertainty
             score_ij = (W_mu - baseline_mean)^2 / (Var(W|x) + baseline_var + eps)
             "Confident anomalous change" gets high score.
          3. Prediction error: absolute residual per node

        Args:
            W_mu: (N, N) causal weight matrix during anomaly
            residual: (N,) prediction residual
            W_var_posterior: (N, N) posterior uncertainty Var(W|x), optional

        Returns:
            rca_scores: (N,) per-node anomaly scores
        """
        # 1. Exogenous anomaly: per-node residual score
        exo_scores = self.residual.mahalanobis_per_node(
            residual.unsqueeze(0)
        ).squeeze(0)  # (N,)

        # 2. Causal change: uncertainty-aware scoring
        #    Denominator = posterior uncertainty + baseline variability
        #    High score = large deviation AND high confidence (low Var(W|x))
        deviation_sq = (W_mu - self.W_baseline_mean) ** 2
        if W_var_posterior is not None:
            denom = W_var_posterior + self.W_baseline_var + 1e-6
        else:
            denom = self.W_baseline_var + 1e-6
        causal_score_mat = deviation_sq / denom  # (N, N)

        causal_in = causal_score_mat.sum(dim=1)   # incoming to node i
        causal_out = causal_score_mat.sum(dim=0)  # outgoing from node j
        causal_scores = causal_in + causal_out

        # 3. Prediction error per node
        pred_error = residual.abs()  # (N,)

        # Normalize each component to comparable scale
        def _safe_normalize(x):
            s = x.sum()
            return x / (s + 1e-8) if s > 1e-8 else x

        rca_scores = (
            lambda_r * _safe_normalize(exo_scores)
            + lambda_w * _safe_normalize(causal_scores)
            + lambda_x * _safe_normalize(pred_error)
        )
        return rca_scores

    # ------------------------------------------------------------------
    #  Evaluation: RCA
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate_rca(
        self,
        abnormal_windows: List[torch.Tensor],
        labels: List[List[int]],
        k_list: List[int] = [1, 3, 5],
        avg_last_k: int = 5,
        n_mc_samples: int = 10,
        carry_filter_state: bool = False,
    ) -> Dict[str, float]:
        """Evaluate RCA using AC@k.

        Args:
            abnormal_windows: list of (T, N) anomaly windows
            labels: list of ground-truth root cause node indices
            k_list: list of k values for AC@k
            avg_last_k: infer_window averaging horizon
            n_mc_samples: infer_window MC samples for posterior uncertainty
            carry_filter_state: carry online state across consecutive windows
        """
        self.eval()
        hit_lists = {k: [] for k in k_list}
        max_k = max(k_list)
        filter_state = None

        for Xw, true_vars in zip(abnormal_windows, labels):
            # Convert per-timestep binary labels (T, N) to root cause node indices
            if isinstance(true_vars, np.ndarray):
                if true_vars.size == 0:
                    continue
                if true_vars.ndim == 2:
                    # (T, N) binary → node indices where any timestep is anomalous
                    true_vars = np.where(true_vars.sum(axis=0) > 0)[0].tolist()
                elif true_vars.ndim == 1:
                    # (N,) binary → node indices
                    true_vars = np.where(true_vars > 0)[0].tolist()
                else:
                    true_vars = true_vars.tolist()
            if not true_vars:
                continue

            Xw = self._to_tensor(Xw).to(self.device)
            if not carry_filter_state:
                filter_state = None
            result = self.infer_window(
                Xw,
                avg_last_k=avg_last_k,
                n_mc_samples=n_mc_samples,
                filter_state=filter_state,
            )
            if carry_filter_state:
                filter_state = result.get("next_filter_state", None)
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
    #  Evaluation: Forecasting-based anomaly detection
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate_detection(
        self,
        abnormal_windows: List[torch.Tensor],
        labels: List[np.ndarray],
        n_mc_samples: int = 1,
        carry_filter_state: bool = False,
    ) -> Dict[str, float]:
        """Evaluate forecasting anomaly detection via predictive NLL."""
        self.eval()
        all_scores = []
        all_targets = []
        filter_state = None

        for Xw, y in zip(abnormal_windows, labels):
            Xw = self._to_tensor(Xw).to(self.device)
            if not carry_filter_state:
                filter_state = None

            result = self.infer_window(
                Xw,
                avg_last_k=1,
                n_mc_samples=n_mc_samples,
                filter_state=filter_state,
            )
            if carry_filter_state:
                filter_state = result.get("next_filter_state", None)

            scores = result["forecast_scores"].detach().cpu().numpy()
            if scores.size == 0:
                continue

            if isinstance(y, torch.Tensor):
                y_arr = y.detach().cpu().numpy()
            else:
                y_arr = np.asarray(y)

            if y_arr.ndim == 2:
                y_any = (y_arr.sum(axis=1) > 0).astype(float)
            elif y_arr.ndim == 1:
                # Node labels only available; treat window as anomalous if any node is anomalous.
                y_any = np.full(Xw.shape[0], float(y_arr.sum() > 0))
            else:
                continue

            n_skipped = int(result.get("n_skipped", Xw.shape[0] - scores.size))
            y_valid = y_any[n_skipped:n_skipped + scores.size]
            if y_valid.size != scores.size:
                m = min(y_valid.size, scores.size)
                y_valid = y_valid[:m]
                scores = scores[:m]
            if y_valid.size == 0:
                continue

            all_scores.append(scores)
            all_targets.append(y_valid)

        if not all_scores:
            return {
                "auroc": 0.0,
                "auprc": 0.0,
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "n": 0,
            }

        s = np.concatenate(all_scores)
        y = np.concatenate(all_targets).astype(float)
        y_hat = (s >= float(self.forecast_threshold.item())).astype(float)

        out = {
            "n": int(s.size),
            "threshold": float(self.forecast_threshold.item()),
        }
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
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        out["precision"] = precision
        out["recall"] = recall
        out["f1"] = 2.0 * precision * recall / (precision + recall + 1e-8)
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
        """Evaluate causal discovery using AUROC/AUPRC.

        Reports two scoring methods and picks the best:
            1. Time-averaged W (empirical causal strength from data)
            2. Learned mu_Z (structural OU prior mean, represents
               the model's belief about edge existence)

        Args:
            causal_struct: (N, N) ground truth adjacency
            normal_sequences: normal data for inference
        """
        self.eval()
        N = self.num_vars

        # --- Score method 1: time-averaged W from data ---
        all_W = []
        for seq in normal_sequences:
            seq = self._to_tensor(seq).to(self.device)[:max_seq_len]
            if seq.shape[0] <= self.lag + 1:
                continue
            result = self.forward_sequence(seq, kl_weight=1.0)
            W = result["W_mean_seq"]  # (m, N, N)
            all_W.append(W.mean(dim=0))

        if not all_W:
            return {"auroc": 0.0, "auprc": 0.0, "f1": 0.0}

        avg_W = torch.stack(all_W).mean(dim=0).cpu().numpy()

        # --- Score method 2: structural signed mean from OU priors ---
        mu_Z_transformed = self._signed_transform(
            self.ou_pos.mu_Z, self.ou_neg.mu_Z
        )[0].detach().cpu().numpy()

        # Evaluate on all entries (including diagonal / self-loops)
        gt = causal_struct.astype(float).ravel()

        def _eval_scores(scores, gt):
            r = {}
            try:
                r["auroc"] = float(roc_auc_score(gt, scores))
            except ValueError:
                r["auroc"] = 0.0
            try:
                r["auprc"] = float(average_precision_score(gt, scores))
            except ValueError:
                r["auprc"] = 0.0
            threshold = np.percentile(scores, 70)
            pred_binary = (scores >= threshold).astype(float)
            r["f1"] = float(f1_score(gt, pred_binary, zero_division=0))
            return r

        # Signed W is mapped to edge strength via absolute value for binary
        # edge discovery metrics against unsigned ground truth.
        w_scores = np.abs(avg_W).ravel()
        z_scores = np.abs(mu_Z_transformed).ravel()
        results_w = _eval_scores(w_scores, gt)
        results_z = _eval_scores(z_scores, gt)

        # Primary: avg_W — data-driven time-averaged inferred W
        results = results_w.copy()
        results["score_method"] = "avg_W"
        results["auroc_W"] = results_w["auroc"]
        results["auroc_muZ"] = results_z["auroc"]

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
        """Full causal summary: time-averaged W mean/var + OU params.

        Returns dict with:
            W_mean: (N, N) time-averaged causal strength
            W_var: (N, N) temporal variability of causal strength
            ou_rho_pos/neg_*: edge-wise mean-reversion summaries
            ou_Q_pos/neg_*: edge-wise structural variability summaries
        """
        self.eval()
        all_W = []

        for seq in normal_sequences:
            seq = self._to_tensor(seq).to(self.device)[:max_seq_len]
            if seq.shape[0] <= self.lag + 1:
                continue
            result = self.forward_sequence(seq, kl_weight=1.0)
            W = result["W_mean_seq"]
            all_W.append(W)

        if not all_W:
            N = self.num_vars
            return {
                "W_mean": torch.zeros(N, N),
                "W_var": torch.zeros(N, N),
                "ou_rho_pos_mean": self.ou_pos.rho.mean().item(),
                "ou_rho_pos_std": self.ou_pos.rho.std().item(),
                "ou_rho_neg_mean": self.ou_neg.rho.mean().item(),
                "ou_rho_neg_std": self.ou_neg.rho.std().item(),
                "ou_Q_pos_mean": self.ou_pos.Q.mean().item(),
                "ou_Q_pos_std": self.ou_pos.Q.std().item(),
                "ou_Q_neg_mean": self.ou_neg.Q.mean().item(),
                "ou_Q_neg_std": self.ou_neg.Q.std().item(),
                "ou_stationary_var_pos_mean": self.ou_pos.stationary_variance.mean().item(),
                "ou_stationary_var_neg_mean": self.ou_neg.stationary_variance.mean().item(),
                "forecast_threshold": float(self.forecast_threshold.item()),
                "forecast_score_mean": float(self.forecast_score_mean.item()),
                "forecast_score_std": float(self.forecast_score_std.item()),
            }

        all_W_cat = torch.cat(all_W, dim=0)  # (total_m, N, N)

        return {
            "W_mean": all_W_cat.mean(dim=0).cpu(),
            "W_var": all_W_cat.var(dim=0).cpu(),
            "ou_rho_pos_mean": self.ou_pos.rho.mean().item(),
            "ou_rho_pos_std": self.ou_pos.rho.std().item(),
            "ou_rho_neg_mean": self.ou_neg.rho.mean().item(),
            "ou_rho_neg_std": self.ou_neg.rho.std().item(),
            "ou_Q_pos_mean": self.ou_pos.Q.mean().item(),
            "ou_Q_pos_std": self.ou_pos.Q.std().item(),
            "ou_Q_neg_mean": self.ou_neg.Q.mean().item(),
            "ou_Q_neg_std": self.ou_neg.Q.std().item(),
            "ou_stationary_var_pos_mean": self.ou_pos.stationary_variance.mean().item(),
            "ou_stationary_var_neg_mean": self.ou_neg.stationary_variance.mean().item(),
            "forecast_threshold": float(self.forecast_threshold.item()),
            "forecast_score_mean": float(self.forecast_score_mean.item()),
            "forecast_score_std": float(self.forecast_score_std.item()),
        }

    # ------------------------------------------------------------------
    #  Model info
    # ------------------------------------------------------------------
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        def _count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            "ou_dynamics_pos": _count(self.ou_pos),
            "ou_dynamics_neg": _count(self.ou_neg),
            "additive_model": _count(self.additive),
            "encoder_pos": _count(self.encoder_pos),
            "encoder_neg": _count(self.encoder_neg),
            "residual_model": _count(self.residual),
            "total": _count(self),
        }
