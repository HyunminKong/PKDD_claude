"""Bayesian z+a 구조 검증 실험 3가지.

1) Prior Robustness  — prior_edge_prob sweep → AUROC 안정성 + calibration
2) a-ℓ Compensation  — lengthscale 고정 vs 학습 → AUROC 비교
3) MLL-drop vs P(z)E[a] — posterior 샘플 MLL-drop과 causal_scores 랭킹 상관

Usage:
    python -m PKDD_claude.gpcdx_rca.verify_bayesian \
        --dataset_name linear --experiment all
"""
from __future__ import annotations

import argparse
import copy
import math
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

from .data import AERCAStyleData
from .gp_granger import GrangerGPVAR


# ======================================================================
#  Helpers
# ======================================================================

def load_dataset(dataset_name: str, window_size: int = 10):
    """Load dataset and return (normal_seqs, causal_struct, N)."""
    opts = {"window_size": window_size, "preprocessing_data": 0}
    loader = AERCAStyleData(dataset_name, opts)
    data = loader.load()
    normal_seqs = AERCAStyleData.extract_normal_windows(data, window_size)
    causal_struct = data.get("causal_struct", None)
    if len(normal_seqs) > 0:
        N = normal_seqs[0].shape[1]
    else:
        N = data["x_ab_list"][0].shape[-1]
    return normal_seqs, causal_struct, N


def train_gp_only(
    normal_seqs,
    N: int,
    device: str,
    window_size: int = 10,
    gp_iters: int = 100,
    prior_edge_prob: float = 0.3,
    feat_dim: int = 8,
    seed: int = 42,
) -> GrangerGPVAR:
    """Train a GrangerGPVAR (GP Phase 1 only, no VGAE)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    gp = GrangerGPVAR(
        num_vars=N, lag=window_size,
        feat_dim=feat_dim, feat_hidden=64,
        device=torch.device(device),
        prior_edge_prob=prior_edge_prob,
    )
    gp.fit(normal_seqs, iters=gp_iters, lr=0.05,
           max_train_samples=500, beta_w=1.0)
    return gp


def evaluate_gp(gp: GrangerGPVAR, causal_struct: np.ndarray) -> Dict[str, float]:
    """Evaluate GP causal discovery: AUROC, AUPRC using P(z)*E[a] and P(z)."""
    gt = causal_struct.astype(float).flatten()
    scores = gp.causal_scores().cpu().numpy().flatten()
    edge_prob = gp.edge_probs.detach().cpu().numpy().flatten()

    results = {}
    try:
        results["auroc"] = float(roc_auc_score(gt, scores))
    except ValueError:
        results["auroc"] = 0.0
    try:
        results["auroc_prob"] = float(roc_auc_score(gt, edge_prob))
    except ValueError:
        results["auroc_prob"] = 0.0
    try:
        results["auprc"] = float(average_precision_score(gt, scores))
    except ValueError:
        results["auprc"] = 0.0
    return results


# ======================================================================
#  Experiment 1: Prior Robustness
# ======================================================================

def experiment_prior_robustness(
    normal_seqs,
    causal_struct: np.ndarray,
    N: int,
    device: str,
    window_size: int = 10,
    gp_iters: int = 100,
    feat_dim: int = 8,
    priors: List[float] = None,
    seeds: List[int] = None,
):
    """prior_edge_prob를 바꿔가며 AUROC 안정성 + calibration 확인."""
    if priors is None:
        priors = [0.1, 0.2, 0.3, 0.5, 0.7]
    if seeds is None:
        seeds = [42, 123, 456]

    gt_flat = causal_struct.astype(float).flatten()
    gt_density = gt_flat.mean()

    print("=" * 70)
    print("  Experiment 1: Prior Robustness (prior_edge_prob sweep)")
    print("=" * 70)
    print(f"  Ground truth density: {gt_density:.2f}")
    print(f"  Priors to test: {priors}")
    print(f"  Seeds: {seeds}")
    print()

    results_all = {}  # prior -> list of dicts per seed

    for prior in priors:
        seed_results = []
        for seed in seeds:
            gp = train_gp_only(
                normal_seqs, N, device, window_size,
                gp_iters=gp_iters, prior_edge_prob=prior,
                feat_dim=feat_dim, seed=seed,
            )
            metrics = evaluate_gp(gp, causal_struct)

            # Calibration: P(z=1) 구간별 실제 edge 비율
            ep = gp.edge_probs.detach().cpu().numpy().flatten()
            # Bin into [0, 0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1.0]
            bins = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
            cal = {}
            for lo, hi in bins:
                mask = (ep >= lo) & (ep < hi) if hi < 1.0 else (ep >= lo) & (ep <= hi)
                if mask.sum() > 0:
                    actual_rate = gt_flat[mask].mean()
                    predicted_mean = ep[mask].mean()
                    cal[f"[{lo:.2f},{hi:.2f})"] = {
                        "count": int(mask.sum()),
                        "predicted": float(predicted_mean),
                        "actual": float(actual_rate),
                    }

            metrics["calibration"] = cal
            metrics["mean_edge_prob"] = float(ep.mean())
            seed_results.append(metrics)
            del gp
            torch.cuda.empty_cache() if device == "cuda" else None

        results_all[prior] = seed_results

    # ── Print summary ──
    print(f"  {'prior':>6} | {'AUROC(P*a)':>12} | {'AUROC(P)':>12} | {'AUPRC':>12} | {'mean P(z)':>10}")
    print("  " + "-" * 65)
    auroc_values = []
    for prior in priors:
        aurocs = [r["auroc"] for r in results_all[prior]]
        aurocs_p = [r["auroc_prob"] for r in results_all[prior]]
        auprcs = [r["auprc"] for r in results_all[prior]]
        mean_ps = [r["mean_edge_prob"] for r in results_all[prior]]
        auroc_values.extend(aurocs)
        print(f"  {prior:>6.2f} | "
              f"{np.mean(aurocs):>5.4f}±{np.std(aurocs):>5.4f} | "
              f"{np.mean(aurocs_p):>5.4f}±{np.std(aurocs_p):>5.4f} | "
              f"{np.mean(auprcs):>5.4f}±{np.std(auprcs):>5.4f} | "
              f"{np.mean(mean_ps):>5.3f}")

    # Robustness check: max - min AUROC across priors
    per_prior_mean = [np.mean([r["auroc"] for r in results_all[p]]) for p in priors]
    spread = max(per_prior_mean) - min(per_prior_mean)
    print(f"\n  AUROC spread (max-min across priors): {spread:.4f}")
    if spread < 0.05:
        print("  ✓ Robust: AUROC varies < 0.05 across priors")
    else:
        print("  ✗ Sensitive: AUROC varies >= 0.05 — prior choice matters")

    # Calibration summary for best prior
    best_prior = priors[np.argmax(per_prior_mean)]
    print(f"\n  Calibration (prior={best_prior}, first seed):")
    cal = results_all[best_prior][0].get("calibration", {})
    for bin_name, info in cal.items():
        print(f"    {bin_name}: predicted={info['predicted']:.3f}, "
              f"actual={info['actual']:.3f}, n={info['count']}")

    return results_all


# ======================================================================
#  Experiment 2: a-ℓ Compensation Test
# ======================================================================

def experiment_lengthscale_compensation(
    normal_seqs,
    causal_struct: np.ndarray,
    N: int,
    device: str,
    window_size: int = 10,
    gp_iters: int = 100,
    feat_dim: int = 8,
    seeds: List[int] = None,
):
    """lengthscale 고정 vs 학습 → AUROC + a 분포 비교."""
    if seeds is None:
        seeds = [42, 123, 456]

    print("\n" + "=" * 70)
    print("  Experiment 2: Lengthscale Compensation (fixed ℓ vs learned ℓ)")
    print("=" * 70)

    results = {"learned": [], "fixed": []}

    for seed in seeds:
        # ── Learned ℓ (default) ──
        gp_learned = train_gp_only(
            normal_seqs, N, device, window_size,
            gp_iters=gp_iters, feat_dim=feat_dim, seed=seed,
        )
        metrics_learned = evaluate_gp(gp_learned, causal_struct)

        # Collect lengthscale and a statistics
        ls_learned = []
        for i in range(N):
            model_i = gp_learned._models[i]
            for j in range(N):
                ls = torch.exp(model_i.kernels[j].log_lengthscale).item()
                ls_learned.append(ls)
        a_learned = gp_learned.edge_strength_mean.detach().cpu().numpy()
        metrics_learned["lengthscales"] = ls_learned
        metrics_learned["a_mean"] = float(a_learned.mean())
        metrics_learned["a_std"] = float(a_learned.std())
        results["learned"].append(metrics_learned)

        # ── Fixed ℓ (freeze log_lengthscale) ──
        torch.manual_seed(seed)
        np.random.seed(seed)
        gp_fixed = GrangerGPVAR(
            num_vars=N, lag=window_size,
            feat_dim=feat_dim, feat_hidden=64,
            device=torch.device(device),
        )
        # Freeze all lengthscale parameters
        for model_i in gp_fixed._models:
            for kernel in model_i.kernels:
                kernel.log_lengthscale.requires_grad = False

        gp_fixed.fit(normal_seqs, iters=gp_iters, lr=0.05,
                      max_train_samples=500, beta_w=1.0)
        metrics_fixed = evaluate_gp(gp_fixed, causal_struct)

        ls_fixed = []
        for i in range(N):
            model_i = gp_fixed._models[i]
            for j in range(N):
                ls = torch.exp(model_i.kernels[j].log_lengthscale).item()
                ls_fixed.append(ls)
        a_fixed = gp_fixed.edge_strength_mean.detach().cpu().numpy()
        metrics_fixed["lengthscales"] = ls_fixed
        metrics_fixed["a_mean"] = float(a_fixed.mean())
        metrics_fixed["a_std"] = float(a_fixed.std())
        results["fixed"].append(metrics_fixed)

        del gp_learned, gp_fixed
        torch.cuda.empty_cache() if device == "cuda" else None

    # ── Print summary ──
    print(f"\n  {'condition':>10} | {'AUROC(P*a)':>12} | {'AUROC(P)':>12} | "
          f"{'AUPRC':>12} | {'ℓ range':>12} | {'E[a]±std':>12}")
    print("  " + "-" * 80)
    for cond in ["learned", "fixed"]:
        aurocs = [r["auroc"] for r in results[cond]]
        aurocs_p = [r["auroc_prob"] for r in results[cond]]
        auprcs = [r["auprc"] for r in results[cond]]
        all_ls = []
        for r in results[cond]:
            all_ls.extend(r["lengthscales"])
        a_means = [r["a_mean"] for r in results[cond]]
        a_stds = [r["a_std"] for r in results[cond]]
        print(f"  {cond:>10} | "
              f"{np.mean(aurocs):>5.4f}±{np.std(aurocs):>5.4f} | "
              f"{np.mean(aurocs_p):>5.4f}±{np.std(aurocs_p):>5.4f} | "
              f"{np.mean(auprcs):>5.4f}±{np.std(auprcs):>5.4f} | "
              f"[{min(all_ls):.2f},{max(all_ls):.2f}] | "
              f"{np.mean(a_means):.3f}±{np.mean(a_stds):.3f}")

    # Check AUROC difference
    auroc_diff = abs(
        np.mean([r["auroc"] for r in results["learned"]])
        - np.mean([r["auroc"] for r in results["fixed"]])
    )
    print(f"\n  AUROC difference (learned - fixed): {auroc_diff:.4f}")
    if auroc_diff < 0.03:
        print("  ✓ Low compensation risk: fixing ℓ barely affects ranking")
    else:
        print("  ✗ Noticeable: ℓ and a may interact — consider fixing ℓ or adding ℓ regularisation")

    return results


# ======================================================================
#  Experiment 3: MLL-drop vs P(z)E[a] Correlation
# ======================================================================

def experiment_mll_drop_correlation(
    normal_seqs,
    causal_struct: np.ndarray,
    N: int,
    device: str,
    window_size: int = 10,
    gp_iters: int = 100,
    feat_dim: int = 8,
    n_posterior_samples: int = 20,
    seed: int = 42,
):
    """Posterior 샘플로 MLL-drop 계산 후, P(z)E[a]와 Spearman 상관 비교."""
    print("\n" + "=" * 70)
    print("  Experiment 3: MLL-drop vs P(z)E[a] Rank Correlation")
    print("=" * 70)

    gp = train_gp_only(
        normal_seqs, N, device, window_size,
        gp_iters=gp_iters, feat_dim=feat_dim, seed=seed,
    )
    gp.eval()

    # ── Reference: P(z)E[a] scores ──
    pz_ea = gp.causal_scores().cpu().numpy()  # (N, N)

    # ── Compute MLL-drop via posterior sampling ──
    # For each posterior sample of (z, a):
    #   For each edge (i, j): compute MLL_full - MLL_without_(i,j)
    print(f"  Sampling {n_posterior_samples} posterior samples for MLL-drop ...")

    raw_feats = gp._raw_feats
    targets = gp._targets
    Z_lists = gp._extract_source_features(raw_feats, differentiable=False)

    mll_drops = np.zeros((n_posterior_samples, N, N))

    for s in range(n_posterior_samples):
        # Sample z, a from posterior
        gp.train()  # enable stochastic sampling
        W_sample = gp.sample_causal_weights().detach()  # (N, N)
        gp.eval()

        for i in range(N):
            # Full MLL for target i with this sample
            mll_full = gp._models[i].mll(
                Z_lists, targets[i], W_sample[i]).item()

            for j in range(N):
                # MLL with edge (i,j) removed
                W_ablated = W_sample[i].clone()
                W_ablated[j] = 0.0
                mll_without = gp._models[i].mll(
                    Z_lists, targets[i], W_ablated).item()

                mll_drops[s, i, j] = mll_full - mll_without

    # ── Statistics ──
    mean_mll_drop = mll_drops.mean(axis=0)  # (N, N)
    std_mll_drop = mll_drops.std(axis=0)    # (N, N)

    # Spearman correlation: P(z)E[a] vs mean MLL-drop
    pz_ea_flat = pz_ea.flatten()
    mll_drop_flat = mean_mll_drop.flatten()
    rho, pval = spearmanr(pz_ea_flat, mll_drop_flat)

    # Also edge_prob alone vs MLL-drop
    ep_flat = gp.edge_probs.detach().cpu().numpy().flatten()
    rho_ep, pval_ep = spearmanr(ep_flat, mll_drop_flat)

    print(f"\n  Spearman correlation:")
    print(f"    P(z)*E[a] vs MLL-drop:   ρ = {rho:.4f}  (p = {pval:.2e})")
    print(f"    P(z) only vs MLL-drop:   ρ = {rho_ep:.4f}  (p = {pval_ep:.2e})")

    if abs(rho) > 0.8:
        print("  ✓ Strong: P(z)E[a] is a good proxy for MLL-drop")
    elif abs(rho) > 0.6:
        print("  △ Moderate: P(z)E[a] partially reflects MLL-drop")
    else:
        print("  ✗ Weak: P(z)E[a] diverges from MLL-drop — consider using MLL-drop directly")

    # AUROC comparison: P(z)E[a] vs mean MLL-drop vs std MLL-drop
    if causal_struct is not None:
        gt_flat = causal_struct.astype(float).flatten()
        try:
            auroc_pzea = roc_auc_score(gt_flat, pz_ea_flat)
        except ValueError:
            auroc_pzea = 0.0
        try:
            auroc_mll = roc_auc_score(gt_flat, mll_drop_flat)
        except ValueError:
            auroc_mll = 0.0
        try:
            # Also try |mean| + std as uncertainty-aware score
            combined = np.abs(mll_drop_flat) + std_mll_drop.flatten()
            auroc_combined = roc_auc_score(gt_flat, combined)
        except ValueError:
            auroc_combined = 0.0

        print(f"\n  AUROC comparison:")
        print(f"    P(z)*E[a]:              {auroc_pzea:.4f}")
        print(f"    Mean MLL-drop:          {auroc_mll:.4f}")
        print(f"    |MLL-drop| + std:       {auroc_combined:.4f}")

    # Print per-edge detail (small N only)
    if N <= 6:
        print(f"\n  Per-edge detail (N={N}):")
        print(f"    {'(i,j)':>6} | {'P(z)*E[a]':>10} | {'MLL-drop μ':>10} | "
              f"{'MLL-drop σ':>10} | {'GT':>4}")
        print("    " + "-" * 55)
        for i in range(N):
            for j in range(N):
                gt_val = int(causal_struct[i, j]) if causal_struct is not None else "?"
                print(f"    ({i},{j})  | {pz_ea[i,j]:>10.4f} | "
                      f"{mean_mll_drop[i,j]:>10.4f} | {std_mll_drop[i,j]:>10.4f} | "
                      f"{gt_val:>4}")

    del gp
    torch.cuda.empty_cache() if device == "cuda" else None

    return {
        "spearman_rho": rho, "spearman_pval": pval,
        "spearman_rho_ep": rho_ep,
        "pz_ea": pz_ea, "mean_mll_drop": mean_mll_drop,
        "std_mll_drop": std_mll_drop,
    }


# ======================================================================
#  Main
# ======================================================================

def main():
    p = argparse.ArgumentParser("Bayesian z+a verification experiments")
    p.add_argument("--dataset_name", type=str, default="linear",
                   choices=["linear", "nonlinear", "lorenz96"])
    p.add_argument("--experiment", type=str, default="all",
                   choices=["all", "prior", "lengthscale", "mll_drop"])
    p.add_argument("--window_size", type=int, default=10)
    p.add_argument("--gp_iters", type=int, default=100)
    p.add_argument("--feat_dim", type=int, default=8)
    p.add_argument("--n_posterior_samples", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset_name}")

    normal_seqs, causal_struct, N = load_dataset(
        args.dataset_name, args.window_size)
    print(f"N={N}, normal sequences: {len(normal_seqs)}")

    if causal_struct is None or not isinstance(causal_struct, np.ndarray):
        print("ERROR: No ground-truth causal structure — need synthetic dataset")
        return

    print(f"Ground truth density: {causal_struct.astype(float).mean():.3f}")
    print()

    if args.experiment in ("all", "prior"):
        experiment_prior_robustness(
            normal_seqs, causal_struct, N, device,
            window_size=args.window_size,
            gp_iters=args.gp_iters,
            feat_dim=args.feat_dim,
        )

    if args.experiment in ("all", "lengthscale"):
        experiment_lengthscale_compensation(
            normal_seqs, causal_struct, N, device,
            window_size=args.window_size,
            gp_iters=args.gp_iters,
            feat_dim=args.feat_dim,
        )

    if args.experiment in ("all", "mll_drop"):
        experiment_mll_drop_correlation(
            normal_seqs, causal_struct, N, device,
            window_size=args.window_size,
            gp_iters=args.gp_iters,
            feat_dim=args.feat_dim,
            n_posterior_samples=args.n_posterior_samples,
            seed=args.seed,
        )

    print("\n" + "=" * 70)
    print("  All experiments complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
