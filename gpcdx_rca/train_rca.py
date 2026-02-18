"""Training script for DyCauseNet.

Usage:
    python -m PKDD_claude.gpcdx_rca.train_rca --dataset_name linear --epochs 50
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from .data import AERCAStyleData
from .model import DyCauseNet


def build_argparser():
    p = argparse.ArgumentParser("DyCauseNet trainer (OU + Additive Granger + RCA)")
    p.add_argument("--dataset_name", type=str, default="linear")
    p.add_argument("--window_size", type=int, default=10)
    # Training
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--kl_anneal_epochs", type=int, default=15)
    p.add_argument("--max_seq_len", type=int, default=500)
    # Model architecture
    p.add_argument("--lag", type=int, default=5)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--feat_dim", type=int, default=8,
                   help="Source feature dimension for causal gating")
    p.add_argument("--encoder_hidden", type=int, default=128)
    p.add_argument("--rank", type=int, default=2,
                   help="Low-rank dimension for residual covariance")
    # OU dynamics
    p.add_argument("--rho_init", type=float, default=0.95,
                   help="Initial mean-reversion speed")
    p.add_argument("--q_init", type=float, default=0.01,
                   help="Initial OU process noise (structural variability)")
    p.add_argument("--mu_z_init", type=float, default=0.0,
                   help="Initial OU mean")
    # Regularization
    p.add_argument("--sparse_weight", type=float, default=0.01,
                   help="L1 sparsity weight on causal weights")
    p.add_argument("--cancel_weight", type=float, default=0.01,
                   help="Signed-channel cancellation penalty")
    p.add_argument("--kl_max", type=float, default=1.0,
                   help="Maximum KL weight")
    p.add_argument("--forecast_quantile", type=float, default=0.99,
                   help="Normal predictive NLL quantile used as anomaly threshold")
    p.add_argument("--rca_avg_last_k", type=int, default=5,
                   help="Average RCA stats over last K filtered steps")
    p.add_argument("--rca_n_mc_samples", type=int, default=10,
                   help="MC samples for posterior Var(W|x) during RCA inference")
    p.add_argument("--rca_carry_filter_state", action="store_true",
                   help="Carry online filtering state across consecutive abnormal windows")
    # Misc
    p.add_argument("--preprocessing_data", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── 1) Load data ──
    loader = AERCAStyleData(args.dataset_name, vars(args))
    data = loader.load()
    normal_seqs = AERCAStyleData.extract_normal_windows(data, args.window_size)
    x_ab_list, labels = AERCAStyleData.extract_abnormal_windows(data)
    causal_struct = data.get("causal_struct", None)

    if not normal_seqs:
        print("No normal sequences found!")
        return

    N = normal_seqs[0].shape[1]
    n_ab = len(x_ab_list) if x_ab_list is not None else 0
    print(f"Dataset: {args.dataset_name}, N={N}, "
          f"normal_seqs={len(normal_seqs)}, "
          f"abnormal_windows={n_ab}")

    # ── 2) Init model ──
    model = DyCauseNet(
        num_vars=N,
        lag=args.lag,
        hidden_dim=args.hidden_dim,
        feat_dim=args.feat_dim,
        encoder_hidden=args.encoder_hidden,
        rank=args.rank,
        rho_init=args.rho_init,
        q_init=args.q_init,
        mu_z_init=args.mu_z_init,
        sparse_weight=args.sparse_weight,
        cancel_weight=args.cancel_weight,
        kl_max=args.kl_max,
        device=torch.device(device),
    )

    # ── Output directory ──
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "outputs", args.dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    # ── 3) Train ──
    t_start = time.time()
    model.fit(
        normal_seqs,
        epochs=args.epochs,
        lr=args.lr,
        kl_anneal_epochs=args.kl_anneal_epochs,
        max_seq_len=args.max_seq_len,
        forecast_quantile=args.forecast_quantile,
        save_dir=out_dir,
    )
    t_train = time.time() - t_start
    print(f"\nTraining time: {t_train:.1f}s")

    # ══════════════════════════════════════════════════════════════════
    #  Evaluation
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  EVALUATION")
    print("=" * 60)

    # ── 4-a) Forecasting anomaly detection ──
    if labels is not None and len(labels) > 0:
        det = model.evaluate_detection(
            x_ab_list,
            labels,
            n_mc_samples=1,
            carry_filter_state=args.rca_carry_filter_state,
        )
        print("\n[Forecast Detection]")
        print(f"  Threshold (train q): {det.get('threshold', 0.0):.4f}  "
              f"(q={args.forecast_quantile:.3f})")
        print(f"  AUROC:  {det.get('auroc', 0.0):.4f}")
        print(f"  AUPRC:  {det.get('auprc', 0.0):.4f}")
        print(f"  F1:     {det.get('f1', 0.0):.4f}")
        print(f"  N:      {det.get('n', 0)}")
    else:
        print("\n[Forecast Detection] No labels found; skipping.")

    # ── 4-b) RCA (AC@k) ──
    if labels is not None and len(labels) > 0:
        t0 = time.time()
        metrics = model.evaluate_rca(
            x_ab_list,
            labels,
            k_list=[1, 3, 5, 10],
            avg_last_k=args.rca_avg_last_k,
            n_mc_samples=args.rca_n_mc_samples,
            carry_filter_state=args.rca_carry_filter_state,
        )
        t_rca = time.time() - t0
        print("\n[RCA]")
        print(f"  Online filtering: True")
        print(f"  avg_last_k={args.rca_avg_last_k}, n_mc={args.rca_n_mc_samples}, "
              f"carry_state={args.rca_carry_filter_state}")
        for k in [1, 3, 5, 10]:
            print(f"  AC@{k}:  {metrics.get(f'ac@{k}', 0):.4f}")
        print(f"  N:     {metrics.get('n', 0)}")
        print(f"  Time:  {t_rca:.2f}s")
    else:
        print("\n[RCA] No labels found; skipping.")

    # ── 4-c) Causal Discovery ──
    if causal_struct is not None and isinstance(causal_struct, np.ndarray) and causal_struct.size > 0:
        cd = model.evaluate_causal_discovery(causal_struct, normal_seqs)
        print("\n[Causal Discovery]")
        print(f"  AUROC:  {cd['auroc']:.4f}  (method: {cd.get('score_method', '?')})")
        print(f"  AUPRC:  {cd['auprc']:.4f}")
        print(f"  F1:     {cd['f1']:.4f}")
        if "auroc_W" in cd:
            print(f"  AUROC (avg_W):     {cd['auroc_W']:.4f}")
            print(f"  AUROC (mu_Z):      {cd['auroc_muZ']:.4f}")

        # Causal summary
        summary = model.causal_summary(normal_seqs)
        W_mean = summary["W_mean"].numpy()
        W_var = summary["W_var"].numpy()
        print(f"\n  OU parameters:")
        print(f"    rho+ mean ± std:      {summary['ou_rho_pos_mean']:.4f} ± {summary['ou_rho_pos_std']:.4f}")
        print(f"    rho- mean ± std:      {summary['ou_rho_neg_mean']:.4f} ± {summary['ou_rho_neg_std']:.4f}")
        print(f"    Q+ mean ± std:        {summary['ou_Q_pos_mean']:.6f} ± {summary['ou_Q_pos_std']:.6f}")
        print(f"    Q- mean ± std:        {summary['ou_Q_neg_mean']:.6f} ± {summary['ou_Q_neg_std']:.6f}")
        print(f"    Forecast threshold:   {summary['forecast_threshold']:.4f}")
        print(f"\n  Causal strength (time-averaged W):")
        for i in range(min(N, 10)):
            row = " ".join(f"{W_mean[i,j]:.3f}" for j in range(min(N, 10)))
            print(f"    [{row}]")
        print(f"\n  Temporal variability (Var[W] across time):")
        for i in range(min(N, 10)):
            row = " ".join(f"{W_var[i,j]:.4f}" for j in range(min(N, 10)))
            print(f"    [{row}]")
        # ── 4-c-vis) Causal matrix visualization ──
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        W_pred = np.abs(W_mean)  # predicted edge strength
        gt_full = causal_struct.astype(float)

        fig, axes = plt.subplots(1, 2, figsize=(5 + 5 * (N > 10), 2.5 + 2.5 * (N > 10)))

        # GT
        im0 = axes[0].imshow(gt_full, cmap="Blues", vmin=0, vmax=1, aspect="equal")
        axes[0].set_title("Ground Truth", fontsize=12, fontweight="bold")
        for i in range(N):
            for j in range(N):
                axes[0].text(j, i, f"{gt_full[i,j]:.0f}",
                             ha="center", va="center", fontsize=max(6, 10 - N // 3),
                             color="white" if gt_full[i, j] > 0.5 else "black")
        axes[0].set_xlabel("Source j")
        axes[0].set_ylabel("Target i")
        axes[0].set_xticks(range(N))
        axes[0].set_yticks(range(N))

        # Predicted |W|
        vmax_pred = max(W_pred.max(), 1e-6)
        im1 = axes[1].imshow(W_pred, cmap="Reds", vmin=0, vmax=vmax_pred, aspect="equal")
        axes[1].set_title(f"Predicted |W|  (AUROC={cd['auroc']:.3f})", fontsize=12, fontweight="bold")
        if N <= 10:
            for i in range(N):
                for j in range(N):
                    axes[1].text(j, i, f"{W_pred[i,j]:.2f}",
                                 ha="center", va="center", fontsize=max(6, 10 - N // 3),
                                 color="white" if W_pred[i, j] > vmax_pred * 0.5 else "black")
        axes[1].set_xlabel("Source j")
        axes[1].set_ylabel("Target i")
        axes[1].set_xticks(range(N))
        axes[1].set_yticks(range(N))
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        fig.suptitle(f"DyCauseNet — {args.dataset_name} (N={N})", fontsize=14, fontweight="bold")
        plt.tight_layout()

        save_path = os.path.join(out_dir, "causal_matrix.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  [Saved] {save_path}")

    else:
        print("\n[Causal Discovery] No ground-truth structure; skipping.")

    # ── 4-d) Model Complexity ──
    pinfo = model.count_parameters()
    print("\n[Model Complexity]")
    for name, count in pinfo.items():
        print(f"  {name}: {count:,}")

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
