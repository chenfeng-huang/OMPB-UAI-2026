from __future__ import annotations

import torch


def mean_pairwise_disagreement(preds: torch.Tensor, tau: float) -> torch.Tensor:
    # preds: [K, batch, pred_len] or [K, batch, pred_len, C]
    k = preds.shape[0]
    if k < 2:
        return torch.tensor(0.0, device=preds.device)
    diffs = preds[:, None, :, :] - preds[None, :, :, :]
    # Reduce over forecast dimensions (pred_len and optional channels)
    reduce_dims = tuple(range(3, diffs.dim()))
    dist2 = torch.sum(diffs**2, dim=reduce_dims) / (tau**2)
    clipped = torch.clamp(dist2, max=1.0)
    mask = torch.triu(torch.ones(k, k, device=preds.device), diagonal=1)
    pairwise = clipped * mask[:, :, None]
    num_pairs = k * (k - 1) / 2.0
    return pairwise.sum() / (num_pairs * preds.shape[1])
