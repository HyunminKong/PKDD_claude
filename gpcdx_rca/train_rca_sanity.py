"""Sanity trainer for DyCauseNet.

Purpose:
    Provide a controlled baseline mode for easy synthetic datasets
    (e.g., linear/nonlinear) where structural recovery should be easier.

Key defaults:
    - Forecasting-based training objective (via DyCauseNet)
    - Optional freezing of OU dynamics (rho/Q) to reduce temporal drift
    - Optional freezing of residual covariance to reduce structure-eating risk
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from .data import AERCAStyleData
from .model import DyCauseNet


def _inv_softplus(x: float, device: torch.device) -> torch.Tensor:
    t = torch.tensor(float(x), device=device)
    return torch.log(torch.expm1(t))


def _apply_sanity_constraints(
    model: DyCauseNet,
    freeze_ou_dynamics: bool = True,
    freeze_residual: bool = True,
    residual_diag_var: float = 1.0,
    fixed_tau: float = 0.5,
):
    """Constrain model freedom for easier identifiability checks."""
    if freeze_ou_dynamics:
        for ou in (model.ou_pos, model.ou_neg):
            ou.raw_q.requires_grad_(False)
            ou.raw_rho.requires_grad_(False)
            ou.tau_init = fixed_tau
            ou.tau_final = fixed_tau
            ou.tau.fill_(fixed_tau)

    if freeze_residual:
        with torch.no_grad():
            model.residual.B.zero_()
            raw_d = _inv_softplus(residual_diag_var, model.device)
            model.residual.raw_d.fill_(raw_d.item())
        for p in model.residual.parameters():
            p.requires_grad_(False)


def build_argparser():
    p = argparse.ArgumentParser("DyCauseNet sanity trainer (forecasting + RCA)")
    p.add_argument("--dataset_name", type=str, default="linear")
    p.add_argument("--window_size", type=int, default=10)
    p.add_argument("--preprocessing_data", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    # Training
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--kl_anneal_epochs", type=int, default=20)
    p.add_argument("--max_seq_len", type=int, default=500)
    p.add_argument("--forecast_quantile", type=float, default=0.99)

    # Model
    p.add_argument("--lag", type=int, default=5)
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--feat_dim", type=int, default=8)
    p.add_argument("--encoder_hidden", type=int, default=64)
    p.add_argument("--rank", type=int, default=1)

    # OU (sanity-friendly defaults: near-static)
    p.add_argument("--rho_init", type=float, default=0.995)
    p.add_argument("--q_init", type=float, default=1e-4)
    p.add_argument("--mu_z_init", type=float, default=-1.0)

    # Regularization
    p.add_argument("--sparse_weight", type=float, default=0.05)
    p.add_argument("--cancel_weight", type=float, default=0.1)
    p.add_argument("--kl_max", type=float, default=0.2)

    # Sanity controls
    p.add_argument("--freeze_ou_dynamics", type=int, default=1,
                   help="1: freeze OU rho/Q and use fixed tau")
    p.add_argument("--freeze_residual", type=int, default=1,
                   help="1: freeze residual covariance to near-identity")
    p.add_argument("--residual_diag_var", type=float, default=1.0)
    p.add_argument("--fixed_tau", type=float, default=0.5)

    # Evaluation
    p.add_argument("--rca_avg_last_k", type=int, default=5)
    p.add_argument("--rca_n_mc_samples", type=int, default=10)
    p.add_argument("--rca_carry_filter_state", action="store_true")
    return p


def main():
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

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
    print(f"Dataset={args.dataset_name}, N={N}, normal={len(normal_seqs)}, abnormal={n_ab}")

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

    _apply_sanity_constraints(
        model,
        freeze_ou_dynamics=bool(args.freeze_ou_dynamics),
        freeze_residual=bool(args.freeze_residual),
        residual_diag_var=args.residual_diag_var,
        fixed_tau=args.fixed_tau,
    )

    print(
        "Sanity constraints:",
        f"freeze_ou={bool(args.freeze_ou_dynamics)},",
        f"freeze_residual={bool(args.freeze_residual)},",
        f"tau={args.fixed_tau}",
    )

    t0 = time.time()
    model.fit(
        normal_seqs,
        epochs=args.epochs,
        lr=args.lr,
        kl_anneal_epochs=args.kl_anneal_epochs,
        max_seq_len=args.max_seq_len,
        forecast_quantile=args.forecast_quantile,
        verbose=True,
    )
    print(f"Training time: {time.time() - t0:.1f}s")

    print("\n" + "=" * 60)
    print("SANITY EVALUATION")
    print("=" * 60)

    if labels is not None and len(labels) > 0:
        det = model.evaluate_detection(
            x_ab_list,
            labels,
            n_mc_samples=1,
            carry_filter_state=args.rca_carry_filter_state,
        )
        print("\n[Forecast Detection]")
        print(f"  Threshold: {det.get('threshold', 0.0):.4f} (q={args.forecast_quantile:.3f})")
        print(f"  AUROC:     {det.get('auroc', 0.0):.4f}")
        print(f"  AUPRC:     {det.get('auprc', 0.0):.4f}")
        print(f"  F1:        {det.get('f1', 0.0):.4f}")
        print(f"  N:         {det.get('n', 0)}")

        rca = model.evaluate_rca(
            x_ab_list,
            labels,
            k_list=[1, 3, 5, 10],
            avg_last_k=args.rca_avg_last_k,
            n_mc_samples=args.rca_n_mc_samples,
            carry_filter_state=args.rca_carry_filter_state,
        )
        print("\n[RCA]")
        for k in [1, 3, 5, 10]:
            print(f"  AC@{k}:     {rca.get(f'ac@{k}', 0.0):.4f}")
        print(f"  N:         {rca.get('n', 0)}")
    else:
        print("\n[Forecast Detection / RCA] labels not found; skipped.")

    if causal_struct is not None and isinstance(causal_struct, np.ndarray) and causal_struct.size > 0:
        cd = model.evaluate_causal_discovery(causal_struct, normal_seqs)
        print("\n[Causal Discovery]")
        print(f"  AUROC:     {cd.get('auroc', 0.0):.4f} ({cd.get('score_method', '?')})")
        print(f"  AUPRC:     {cd.get('auprc', 0.0):.4f}")
        print(f"  F1:        {cd.get('f1', 0.0):.4f}")
        if "auroc_W" in cd:
            print(f"  AUROC(avgW): {cd['auroc_W']:.4f}")
            print(f"  AUROC(muZ):  {cd['auroc_muZ']:.4f}")
    else:
        print("\n[Causal Discovery] causal_struct not found; skipped.")

    summary = model.causal_summary(normal_seqs)
    print("\n[Summary]")
    print(f"  Forecast threshold: {summary.get('forecast_threshold', 0.0):.4f}")
    print(f"  rho+ mean:          {summary.get('ou_rho_pos_mean', 0.0):.4f}")
    print(f"  rho- mean:          {summary.get('ou_rho_neg_mean', 0.0):.4f}")
    print(f"  Q+ mean:            {summary.get('ou_Q_pos_mean', 0.0):.6f}")
    print(f"  Q- mean:            {summary.get('ou_Q_neg_mean', 0.0):.6f}")

    pinfo = model.count_parameters()
    print("\n[Model Complexity]")
    for name, count in pinfo.items():
        print(f"  {name}: {count:,}")

    print("\nDone.")


if __name__ == "__main__":
    main()

