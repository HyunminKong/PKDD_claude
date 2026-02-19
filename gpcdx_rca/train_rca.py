"""Training script for DyCauseNet v3 (bypass-blocked).

Usage:
    python -m PKDD_claude.gpcdx_rca.train_rca --dataset_name linear --epochs 300
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
    p = argparse.ArgumentParser("DyCauseNet v3 trainer")
    p.add_argument("--dataset_name", type=str, default="linear")
    p.add_argument("--window_size", type=int, default=10)
    # Training
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--kl_anneal_epochs", type=int, default=15)
    p.add_argument("--max_seq_len", type=int, default=500)
    # Model architecture
    p.add_argument("--lag", type=int, default=5)
    p.add_argument("--basis_hidden", type=int, default=32,
                   help="Basis MLP residual hidden dim")
    p.add_argument("--gru_hidden", type=int, default=128,
                   help="InferenceNet GRU hidden dim")
    p.add_argument("--bottleneck_dim", type=int, default=8,
                   help="InferenceNet temporal bottleneck dim")
    p.add_argument("--emb_dim", type=int, default=16,
                   help="Embedding dim for (j, i, k)")
    p.add_argument("--mlp_hidden", type=int, default=64,
                   help="InferenceNet MLP hidden dim")
    p.add_argument("--var_hidden", type=int, default=32,
                   help="VarHead hidden dim")
    # OU dynamics
    p.add_argument("--rho_init", type=float, default=0.95)
    p.add_argument("--q_init", type=float, default=0.01)
    # Loss weights
    p.add_argument("--kl_max", type=float, default=1.0,
                   help="Maximum KL weight β")
    p.add_argument("--sparse_weight", type=float, default=1e-3,
                   help="Group sparsity weight λ")
    p.add_argument("--exog_weight", type=float, default=0.1,
                   help="Exogenous independence weight η")
    # Evaluation
    p.add_argument("--forecast_quantile", type=float, default=0.99)
    p.add_argument("--rca_avg_last_k", type=int, default=5)
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
    print(f"Basis: identity+MLP(h={args.basis_hidden}), "
          f"Bottleneck: {args.bottleneck_dim}, "
          f"Loss weights: kl_max={args.kl_max}, "
          f"sparse={args.sparse_weight}, exog={args.exog_weight}")

    # ── 2) Init model ──
    model = DyCauseNet(
        num_vars=N,
        lag=args.lag,
        basis_hidden=args.basis_hidden,
        gru_hidden=args.gru_hidden,
        bottleneck_dim=args.bottleneck_dim,
        emb_dim=args.emb_dim,
        mlp_hidden=args.mlp_hidden,
        var_hidden=args.var_hidden,
        rho_init=args.rho_init,
        q_init=args.q_init,
        kl_max=args.kl_max,
        sparse_weight=args.sparse_weight,
        exog_weight=args.exog_weight,
        device=torch.device(device),
    )

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
        det = model.evaluate_detection(x_ab_list, labels)
        print("\n[Forecast Detection]")
        print(f"  Threshold (train q): {det.get('threshold', 0.0):.4f}")
        print(f"  AUROC:  {det.get('auroc', 0.0):.4f}")
        print(f"  AUPRC:  {det.get('auprc', 0.0):.4f}")
        print(f"  F1:     {det.get('f1', 0.0):.4f}")
        print(f"  N:      {det.get('n', 0)}")

    # ── 4-b) RCA (AC@k) ──
    if labels is not None and len(labels) > 0:
        t0 = time.time()
        metrics = model.evaluate_rca(
            x_ab_list, labels,
            k_list=[1, 3, 5, 10],
            avg_last_k=args.rca_avg_last_k,
        )
        t_rca = time.time() - t0
        print("\n[RCA]")
        for k in [1, 3, 5, 10]:
            print(f"  AC@{k}:  {metrics.get(f'ac@{k}', 0):.4f}")
        print(f"  N:     {metrics.get('n', 0)}")
        print(f"  Time:  {t_rca:.2f}s")

    # ── 4-c) Causal Discovery ──
    if causal_struct is not None and isinstance(causal_struct, np.ndarray) and causal_struct.size > 0:
        cd = model.evaluate_causal_discovery(causal_struct, normal_seqs)
        print("\n[Causal Discovery]")
        print(f"  AUROC:  {cd['auroc']:.4f}  (method: {cd.get('score_method', '?')})")
        print(f"  AUPRC:  {cd['auprc']:.4f}")
        print(f"  F1:     {cd['f1']:.4f}")

        summary = model.causal_summary(normal_seqs)
        A_mean = summary["A_mean"].numpy()
        A_var = summary["A_var"].numpy()
        print(f"\n  OU parameters:")
        print(f"    rho:                {summary['ou_rho']:.4f}")
        print(f"    Q:                  {summary['ou_Q']:.6f}")
        print(f"    Forecast threshold: {summary['forecast_threshold']:.4f}")
        print(f"\n  Causal strength (time-averaged Ā = sqrt(Σ_k A_mu²)):")
        for i in range(min(N, 10)):
            row = " ".join(f"{A_mean[i,j]:.3f}" for j in range(min(N, 10)))
            print(f"    [{row}]")

        # ── Visualization ──
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        gt_full = causal_struct.astype(float)
        fig, axes = plt.subplots(1, 2, figsize=(5 + 5 * (N > 10), 2.5 + 2.5 * (N > 10)))

        im0 = axes[0].imshow(gt_full, cmap="Blues", vmin=0, vmax=1, aspect="equal")
        axes[0].set_title("Ground Truth", fontsize=12, fontweight="bold")
        for i in range(N):
            for j in range(N):
                axes[0].text(j, i, f"{gt_full[i,j]:.0f}",
                             ha="center", va="center", fontsize=max(6, 10 - N // 3),
                             color="white" if gt_full[i, j] > 0.5 else "black")
        axes[0].set_xlabel("Source i")
        axes[0].set_ylabel("Target j")
        axes[0].set_xticks(range(N))
        axes[0].set_yticks(range(N))

        vmax_pred = max(A_mean.max(), 1e-6)
        im1 = axes[1].imshow(A_mean, cmap="Reds", vmin=0, vmax=vmax_pred, aspect="equal")
        axes[1].set_title(f"Predicted Ā  (AUROC={cd['auroc']:.3f})", fontsize=12, fontweight="bold")
        if N <= 10:
            for i in range(N):
                for j in range(N):
                    axes[1].text(j, i, f"{A_mean[i,j]:.2f}",
                                 ha="center", va="center", fontsize=max(6, 10 - N // 3),
                                 color="white" if A_mean[i, j] > vmax_pred * 0.5 else "black")
        axes[1].set_xlabel("Source i")
        axes[1].set_ylabel("Target j")
        axes[1].set_xticks(range(N))
        axes[1].set_yticks(range(N))
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        fig.suptitle(f"DyCauseNet v3 — {args.dataset_name} (N={N})", fontsize=14, fontweight="bold")
        plt.tight_layout()

        save_path = os.path.join(out_dir, "causal_matrix.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  [Saved] {save_path}")

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
