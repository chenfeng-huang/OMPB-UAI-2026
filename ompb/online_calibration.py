from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from .bound import gamma_bound
from .disagreement import mean_pairwise_disagreement

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(it, **kwargs):
        return it


def _compute_gibbs_risk(
    head,
    base: torch.Tensor,
    y: torch.Tensor,
    k: int,
) -> torch.Tensor:
    preds = head.forward_samples(base, k)
    mse = (preds - y.unsqueeze(0)) ** 2
    return mse.mean()


def _compute_disagreement(
    head,
    base: torch.Tensor,
    k: int,
    tau: float,
) -> torch.Tensor:
    preds = head.forward_samples(base, k)
    return mean_pairwise_disagreement(preds, tau)


def _consistency_loss(head, backbone, w_prev: torch.Tensor, w_curr: torch.Tensor, device: str) -> torch.Tensor:
    base_prev = backbone.predict_batch(w_prev.unsqueeze(0)).to(device)
    base_curr = backbone.predict_batch(w_curr.unsqueeze(0)).to(device)
    y_prev = head.forward_mean(base_prev)[0]
    y_curr = head.forward_mean(base_curr)[0]
    if y_prev.shape[0] < 2:
        return torch.tensor(0.0, device=device)
    return torch.mean((y_prev[1:] - y_curr[:-1]) ** 2)


def run_online_calibration(
    backbone,
    head,
    source_buffer: torch.Tensor,
    source_labels: torch.Tensor,
    target_dataset,
    config: Dict,
    constants: Dict,
    device: str,
) -> List[Dict]:
    def _is_torch_backbone(bb) -> bool:
        # ARIMA/KRR are sklearn/statsmodels-based and operate on CPU/numpy.
        return hasattr(bb, "model") and isinstance(getattr(bb, "model"), torch.nn.Module)

    def _prefetch_windows(ds, dev: str):
        """
        Fast-path: materialize all sliding windows once on the target device.
        This avoids per-step numpy slicing + torch.from_numpy in the online loop.
        """
        if not (hasattr(ds, "data") and hasattr(ds, "seq_len") and hasattr(ds, "pred_len")):
            return None
        try:
            seq_len = int(ds.seq_len)
            pred_len = int(ds.pred_len)
            data = ds.data  # numpy float32 [T, F]
        except Exception:
            return None

        # Build sliding windows.
        # For a 2D tensor [T, F], `unfold(0, size, 1)` returns [N, F, size],
        # so we permute to [N, size, F].
        x_all = torch.from_numpy(data).to(dev)  # [T, F]
        win = x_all.unfold(0, seq_len + pred_len, 1)  # [N, F, seq_len+pred_len]
        x_win = win[:, :, :seq_len].permute(0, 2, 1).contiguous()  # [N, seq_len, F]
        y_win = win[:, :, seq_len : seq_len + pred_len].permute(0, 2, 1).contiguous()  # [N, pred_len, F]
        return x_win, y_win

    k = int(config["K"])
    buffer_l = int(config["buffer_L"])
    cal_steps = int(config["cal_steps_J"])
    cal_lr = float(config["cal_lr"])
    tau = float(config["tau"])
    lambda_tr = float(config["lambda_tr"])
    lambda_cons = float(config["lambda_cons"])
    update_rate = float(config.get("update_rate", 1.0))
    update_rate = max(0.0, min(1.0, update_rate))
    lambda_sup = float(config.get("lambda_sup", 0.0))
    src_batch_size = int(config.get("src_batch_size", 64))

    optimizer = torch.optim.Adam(head.parameters(), lr=cal_lr)
    logs: List[Dict] = []
    target_buffer: List[torch.Tensor] = []

    torch_backbone = _is_torch_backbone(backbone)
    buffer_device = device if torch_backbone else "cpu"
    source_buffer = source_buffer.to(buffer_device)
    source_labels = source_labels.to(buffer_device)

    prefetch = config.get("prefetch_target", 1)
    x_all = None
    y_all = None
    if torch_backbone and str(prefetch) not in ("0", "false", "False", "no", "No"):
        prefetched = _prefetch_windows(target_dataset, buffer_device)
        if prefetched is not None:
            x_all, y_all = prefetched

    ring = None
    ring_valid = 0
    ring_pos = 0
    if torch_backbone:
        if x_all is not None:
            _, seq_len, n_feat = x_all.shape
            ring = torch.empty((buffer_l, seq_len, n_feat), device=buffer_device, dtype=x_all.dtype)

    max_t = config.get("max_t")
    total_t = len(target_dataset)
    if max_t is not None:
        total_t = min(total_t, int(max_t))

    show_pbar = config.get("progress", 1)
    it = range(total_t)
    if str(show_pbar) not in ("0", "false", "False", "no", "No"):
        it = tqdm(it, total=total_t, desc="OMPB online", leave=False)

    for idx in it:
        if x_all is not None and y_all is not None:
            x_t = x_all[idx]
            y_t = y_all[idx]
        else:
            x_t, y_t = target_dataset[idx]  # dataset yields CPU tensors
        x_buf = x_t.to(buffer_device) if torch_backbone else x_t
        if ring is not None:
            ring[ring_pos].copy_(x_buf)
            ring_pos = (ring_pos + 1) % buffer_l
            ring_valid = min(buffer_l, ring_valid + 1)
        else:
            target_buffer.append(x_buf)
            if len(target_buffer) > buffer_l:
                target_buffer.pop(0)

        # IMPORTANT (causality / no leakage):
        # Produce the window prediction *before* any gradient update that can use y_t.
        head.eval()
        x_eval = x_t.to(device) if torch_backbone else x_t
        with torch.no_grad():
            base_eval = backbone.predict_batch(x_eval.unsqueeze(0)).to(device)
            y_pred = head.forward_mean(base_eval)[0].detach().cpu().numpy()
        y_true = y_t.detach().cpu().numpy()
        mae = float(np.mean(np.abs(y_true - y_pred)))
        mse = float(np.mean((y_true - y_pred) ** 2))

        base_sup = None
        y_sup = None
        if lambda_sup > 0.0:
            # Reuse the already-computed base forecast for this window.
            base_sup = base_eval
            y_sup = y_t.unsqueeze(0).to(device)

        prev_params = head.snapshot()
        dis_hat_value = 0.0
        risk_value = 0.0
        gamma_value = 0.0
        for _ in range(cal_steps):
            head.train()
            m = int(len(source_buffer))
            bs = min(int(src_batch_size), m)
            if buffer_device != "cpu":
                # GPU-friendly sampling (avoids numpy + CPU indexing overhead).
                perm = torch.randperm(m, device=source_buffer.device)[:bs]
                src_x = source_buffer.index_select(0, perm)
                src_y = source_labels.index_select(0, perm)
            else:
                src_idx = np.random.choice(m, size=bs, replace=False)
                src_x = source_buffer[src_idx]
                src_y = source_labels[src_idx]

            params_before = head.snapshot()
            base_src = backbone.predict_batch(src_x).to(device)
            risk = _compute_gibbs_risk(head, base_src, src_y, k)

            kl_prior = head.kl_to_prior()
            gamma = gamma_bound(kl_prior, constants["V_hat"], constants["bar_c"], constants["kappa"], constants["m"])

            if ring is not None:
                # Build contiguous view of last `ring_valid` entries in chronological order.
                if ring_valid < buffer_l:
                    tgt_x = ring[:ring_valid]
                else:
                    tgt_x = torch.cat([ring[ring_pos:], ring[:ring_pos]], dim=0)
                tgt_x = tgt_x.to(device)
            else:
                # If buffer_device is CUDA, this stack stays on GPU (no per-step CPU->GPU copies).
                tgt_x = torch.stack(target_buffer, dim=0).to(device)
            base_tgt = backbone.predict_batch(tgt_x).to(device)
            dis_src = _compute_disagreement(head, base_src, k, tau)
            dis_tgt = _compute_disagreement(head, base_tgt, k, tau)
            dis_hat = torch.abs(dis_src - dis_tgt)


        

            sup_loss = torch.tensor(0.0, device=device)
            if base_sup is not None and y_sup is not None:
                sup_pred = head.forward_mean(base_sup)
                sup_loss = torch.mean((sup_pred - y_sup) ** 2)

            loss = risk + gamma + 0.5 * dis_hat + lambda_sup * sup_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Optional damping: blend updated params with previous params to reduce harmful drift.
            if update_rate < 1.0:
                with torch.no_grad():
                    head.mu_w.copy_((1.0 - update_rate) * params_before.mu_w + update_rate * head.mu_w)
                    head.log_sigma_w.copy_((1.0 - update_rate) * params_before.log_sigma_w + update_rate * head.log_sigma_w)
                    head.mu_b.copy_((1.0 - update_rate) * params_before.mu_b + update_rate * head.mu_b)
                    head.log_sigma_b.copy_((1.0 - update_rate) * params_before.log_sigma_b + update_rate * head.log_sigma_b)
                    if hasattr(head, "alpha_logit"):
                        head.alpha_logit.copy_((1.0 - update_rate) * params_before.alpha_logit + update_rate * head.alpha_logit)

            dis_hat_value = float(dis_hat.detach().cpu().item())
            risk_value = float(risk.detach().cpu().item())
            gamma_value = float(gamma.detach().cpu().item())

        kl_prior_val = float(head.kl_to_prior().detach().cpu().item())
        kl_prev_val = float(head.kl_to_prev_posterior(prev_params).detach().cpu().item())
        cert_val = risk_value + gamma_value + 0.5 * dis_hat_value

        entry: Dict = {
                "t": idx,
                "mae": mae,
                "mse": mse,
                "risk_hat": risk_value,
                "gamma": gamma_value,
                "dis_hat": dis_hat_value,
                "cert_pb": cert_val,
                "kl_prior": kl_prior_val,
                "kl_prev": kl_prev_val,
        }
        if config.get("store_preds", False):
            entry["y_pred"] = y_pred  # numpy [pred_len, F]
            entry["y_true"] = y_true  # numpy [pred_len, F]
        logs.append(entry)
    return logs
