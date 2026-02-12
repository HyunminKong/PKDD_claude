import torch


def normalize_adjacency(A: torch.Tensor) -> torch.Tensor:
    """Symmetric normalization: Ã = D^{-1/2}AD^{-1/2} (Section 4.3).

    Note: self-loops are NOT added explicitly — the Granger causality matrix
    already has non-zero diagonal (self-causality μ_{jj}).
    """
    d = A.sum(-1).clamp(min=1e-8)
    D_inv_sqrt = torch.diag(torch.pow(d, -0.5))
    return D_inv_sqrt @ A @ D_inv_sqrt


def wasserstein2_gaussian(mu1, sigma1, mu2, sigma2):
    """Squared 2-Wasserstein distance between diagonal Gaussians.

    W2^2 = ||mu1 - mu2||^2 + ||sigma1 - sigma2||^2
    """
    return ((mu1 - mu2) ** 2).sum() + ((sigma1 - sigma2) ** 2).sum()


def topk_hit_rate(pred_scores, true_indices, k=1):
    """Compute top-k hit (1 if any of true_indices in top-k of pred_scores)."""
    k = min(k, pred_scores.numel())
    topk = torch.topk(pred_scores, k=k).indices.tolist()
    truth = set(int(i) for i in true_indices)
    return 1.0 if any(i in truth for i in topk) else 0.0
