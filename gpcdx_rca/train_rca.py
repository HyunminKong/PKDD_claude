from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .data import AERCAStyleData
from .model import GPCDX_RCA


def build_argparser():
    p = argparse.ArgumentParser("GPCDX-RCA trainer (Deep Kernel + End-to-End)")
    p.add_argument("--dataset_name", type=str, default="msds")
    p.add_argument("--window_size", type=int, default=10)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    # GP arguments
    p.add_argument("--gp_iters", type=int, default=20)
    p.add_argument("--gp_lr", type=float, default=0.05)
    p.add_argument("--gp_max_train", type=int, default=500,
                   help="Max training samples per GP pair (sub-sampled for efficiency)")
    # Deep Kernel arguments
    p.add_argument("--feat_dim", type=int, default=16,
                   help="Feature extractor output dimension (d_feat for Deep Kernel)")
    p.add_argument("--feat_hidden", type=int, default=64,
                   help="Feature extractor hidden dimension")
    # VGAE arguments
    p.add_argument("--vgae_hidden", type=int, default=64)
    p.add_argument("--vgae_latent", type=int, default=8)
    p.add_argument("--beta_kl", type=float, default=1.0)
    # End-to-end arguments
    p.add_argument("--sparse_weight", type=float, default=0.01,
                   help="L1 sparsity weight on causal adjacency (end_to_end mode)")
    # Training mode
    p.add_argument("--training_mode", type=str, default="end_to_end",
                   choices=["two_stage", "joint", "end_to_end"],
                   help="two_stage: pre-compute GP then train VGAE; "
                        "joint: on-the-fly GP (no gradient); "
                        "end_to_end: jointly optimise feature extractor + VGAE")
    # Misc
    p.add_argument("--preprocessing_data", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--causal_quantile", type=float, default=0.70)
    return p


def main():
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Auto-select GPU if available
    auto_device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = auto_device
    print(f"Using device: {args.device}")

    # ── 1) Load data ──
    loader = AERCAStyleData(args.dataset_name, vars(args))
    data = loader.load()
    normal_seqs = AERCAStyleData.extract_normal_windows(data, args.window_size)
    x_ab_list, labels = AERCAStyleData.extract_abnormal_windows(data)

    # Ground-truth causal structure (synthetic datasets only)
    causal_struct = data.get("causal_struct", None)

    # ── 2) Init model ──
    if len(normal_seqs) > 0:
        inferred_N = normal_seqs[0].shape[1]
    else:
        inferred_N = x_ab_list[0].shape[-1]

    model = GPCDX_RCA(
        num_vars=inferred_N,
        window_size=args.window_size,
        device=args.device,
        gp_iters=args.gp_iters,
        gp_lr=args.gp_lr,
        gp_max_train=args.gp_max_train,
        feat_dim=args.feat_dim,
        feat_hidden=args.feat_hidden,
        vgae_hidden=args.vgae_hidden,
        vgae_latent=args.vgae_latent,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta_kl=args.beta_kl,
        sparse_weight=args.sparse_weight,
    )

    # ── 3) Train ──
    print(f"Training mode: {args.training_mode}")
    model.fit(normal_seqs, training_mode=args.training_mode)

    # ══════════════════════════════════════════════════════════════════
    #  Evaluation
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  EVALUATION")
    print("=" * 60)

    # ── 4-a) RCA (AC@k) ──
    if labels is not None:
        metrics = model.evaluate_rca(x_ab_list, labels, k_list=(1, 3, 5, 10))
        print("\n[RCA]")
        print(f"  AC@1:  {metrics.get('ac@1', 0):.4f}")
        print(f"  AC@3:  {metrics.get('ac@3', 0):.4f}")
        print(f"  AC@5:  {metrics.get('ac@5', 0):.4f}")
        print(f"  AC@10: {metrics.get('ac@10', 0):.4f}")
        print(f"  N:     {metrics.get('n', 0)}")
    else:
        print("\n[RCA] No labels found; skipping.")

    # ── 4-b) Causal Discovery ──
    if causal_struct is not None and isinstance(causal_struct, np.ndarray) and causal_struct.size > 0:
        cd = model.evaluate_causal_discovery(
            causal_struct,
            test_sequences=[arr for arr in x_ab_list] if x_ab_list is not None else None,
            causal_quantile=args.causal_quantile,
        )
        print("\n[Causal Discovery]")
        print(f"  AUROC:            {cd['auroc']:.4f}")
        print(f"  AUPRC:            {cd['auprc']:.4f}")
        print(f"  F1:               {cd['f1']:.4f}")
        print(f"  Hamming Distance: {cd['hamming']:.4f}")
    else:
        print("\n[Causal Discovery] No ground-truth causal structure; skipping.")

    # ── 4-c) Model Complexity ──
    pinfo = model.count_parameters()
    ginfo = model.estimate_gflops()
    print("\n[Model Complexity]")
    print(f"  VGAE params:           {pinfo['vgae_params']:,}")
    print(f"  Feature extractor:     {pinfo['feat_extractor_params']:,}")
    print(f"  GP per-pair params:    {pinfo['gp_params_per_pair']} x {inferred_N}^2 = {pinfo['gp_pair_total']:,}")
    print(f"  GP total (feat+pairs): {pinfo['gp_total']:,}")
    print(f"  Total params:          {pinfo['total']:,}")
    print(f"  VGAE+FE GFLOPs:       {ginfo['gflops']:.6f}")
    print(f"  VGAE+FE MFLOPs:       {ginfo['mflops']:.4f}")

    # ── 4-d) Latency ──
    if x_ab_list is not None and len(x_ab_list) > 0:
        lat = model.measure_latency(x_ab_list, n_warmup=3)
        print("\n[Latency]")
        print(f"  Total:           {lat['total_sec']:.2f} s  ({lat['n_windows']} windows)")
        print(f"  Per window:      {lat['per_window_ms']:.2f} ms")
    else:
        print("\n[Latency] No test windows; skipping.")

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
