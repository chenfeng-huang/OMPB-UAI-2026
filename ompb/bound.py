from __future__ import annotations

from typing import Dict

import numpy as np
import torch


def compute_variance_proxy_constants(
    source_dataset,
    backbone,
    head,
    alpha: float,
    beta: float,
    beta_c: float,
    delta: float,
    c0: float,
    device: str,
) -> Dict[str, float]:
    residuals = []
    for x, y in source_dataset:
        x = x.unsqueeze(0).to(device)
        base = backbone.predict_batch(x).to(device)
        y_hat = head.forward_mean(base)[0]
        # Support both scalar targets ([pred_len]) and multivariate ([pred_len, C]).
        # We treat each element residual as one sample for the variance proxy.
        r = (y.to(device) - y_hat).detach().reshape(-1)
        residuals.extend([float(v.item()) for v in r])

    s2 = 1.0
    v_hat = 0.0
    max_s2 = 0.0
    for r in residuals:
        s2 = beta * s2 + (1.0 - beta) * (r**2)
        v_hat += alpha * (s2**4)
        if s2 > max_s2:
            max_s2 = s2

    m = max(1, len(residuals))
    bar_c = beta_c * max_s2
    kappa = float(np.log(c0 * np.sqrt(m) / delta))
    return {
        "V_hat": float(v_hat),
        "bar_c": float(bar_c),
        "kappa": float(kappa),
        "m": float(m),
    }


def gamma_bound(kl: torch.Tensor, V_hat: float, bar_c: float, kappa: float, m: float) -> torch.Tensor:
    term = kl + kappa
    return torch.sqrt(2.0 * V_hat * term / (m**2)) + (bar_c / m) * term
