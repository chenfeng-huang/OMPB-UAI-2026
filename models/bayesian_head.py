from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


@dataclass
class PosteriorParams:
    mu_w: torch.Tensor
    log_sigma_w: torch.Tensor
    mu_b: torch.Tensor
    log_sigma_b: torch.Tensor
    alpha_logit: torch.Tensor


class BayesianLinearHead(nn.Module):
    def __init__(
        self,
        pred_len: int,
        out_channels: int,
        sigma0: float,
        alpha_init: float = -6.0,
        alpha_prior: float = -6.0,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.out_channels = int(out_channels)
        self.sigma0 = sigma0
        # Redesign: learn a *residual* correction on top of the backbone forecast.
        #
        # Let base forecast be z (shape [B, pred_len]). The head outputs:
        #   y = z + s * (z @ ΔW^T + Δb)
        # where s = sigmoid(alpha_logit) ∈ (0,1).
        #
        # Initialization makes this "do no harm":
        #   ΔW = 0, Δb = 0, alpha_logit ≪ 0  =>  y ≈ z
        #
        # Prior is centered at ΔW=0, Δb=0, alpha_logit=alpha_prior (favoring s≈0 by default).
        if self.out_channels == 1:
            self.register_buffer("prior_mu_dw", torch.zeros(pred_len, pred_len))
            self.register_buffer("prior_mu_db", torch.zeros(pred_len))
        else:
            # Per-channel independent correction (shared gate).
            self.register_buffer("prior_mu_dw", torch.zeros(self.out_channels, pred_len, pred_len))
            self.register_buffer("prior_mu_db", torch.zeros(self.out_channels, pred_len))
        self.register_buffer("prior_alpha_logit", torch.tensor(float(alpha_prior)))

        if self.out_channels == 1:
            self.mu_w = nn.Parameter(torch.zeros(pred_len, pred_len))  # ΔW mean
            self.log_sigma_w = nn.Parameter(torch.zeros(pred_len, pred_len))  # ΔW log-std
            self.mu_b = nn.Parameter(torch.zeros(pred_len))  # Δb mean
            self.log_sigma_b = nn.Parameter(torch.zeros(pred_len))  # Δb log-std
        else:
            self.mu_w = nn.Parameter(torch.zeros(self.out_channels, pred_len, pred_len))  # [C, O, I]
            self.log_sigma_w = nn.Parameter(torch.zeros(self.out_channels, pred_len, pred_len))
            self.mu_b = nn.Parameter(torch.zeros(self.out_channels, pred_len))  # [C, O]
            self.log_sigma_b = nn.Parameter(torch.zeros(self.out_channels, pred_len))
        self.alpha_logit = nn.Parameter(torch.tensor(float(alpha_init)))

    def sample(self, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.out_channels == 1:
            eps_w = torch.randn(k, self.pred_len, self.pred_len, device=self.mu_w.device)
            eps_b = torch.randn(k, self.pred_len, device=self.mu_b.device)
        else:
            eps_w = torch.randn(k, self.out_channels, self.pred_len, self.pred_len, device=self.mu_w.device)
            eps_b = torch.randn(k, self.out_channels, self.pred_len, device=self.mu_b.device)
        sigma_w = torch.exp(self.log_sigma_w)
        sigma_b = torch.exp(self.log_sigma_b)
        dw = self.mu_w.unsqueeze(0) + eps_w * sigma_w.unsqueeze(0)
        db = self.mu_b.unsqueeze(0) + eps_b * sigma_b.unsqueeze(0)
        # Gate is deterministic (not sampled) for stability.
        s = torch.sigmoid(self.alpha_logit).detach()
        return dw * s, db * s

    def forward_mean(self, z: torch.Tensor) -> torch.Tensor:
        s = torch.sigmoid(self.alpha_logit)
        if self.out_channels == 1:
            # z: [B, pred_len]
            return z + s * (z @ self.mu_w.t() + self.mu_b)
        # z: [B, pred_len, C]
        # per-channel: out[b, o, c] = z[b, o, c] + s * (sum_i z[b,i,c]*mu_w[c,o,i] + mu_b[c,o])
        corr = torch.einsum("bic,coi->boc", z, self.mu_w) + self.mu_b.transpose(0, 1).unsqueeze(0)
        return z + s * corr

    def forward_samples(self, z: torch.Tensor, k: int) -> torch.Tensor:
        dw, db = self.sample(k)
        if self.out_channels == 1:
            # z: [B, pred_len] -> preds: [K, B, pred_len]
            preds = z.unsqueeze(0) + torch.einsum("kij,bj->kbi", dw, z) + db[:, None, :]
            return preds
        # z: [B, pred_len, C] -> preds: [K, B, pred_len, C]
        corr = torch.einsum("kcoi,bic->kboc", dw, z) + db[:, None, :, :].transpose(2, 3)
        return z.unsqueeze(0) + corr

    def kl_to_prior(self) -> torch.Tensor:
        sigma0 = torch.tensor(self.sigma0, device=self.mu_w.device)
        sigma_w = torch.exp(self.log_sigma_w)
        sigma_b = torch.exp(self.log_sigma_b)
        prior_mu_w = self.prior_mu_dw.to(self.mu_w.device)
        prior_mu_b = self.prior_mu_db.to(self.mu_b.device)
        kl_w = 0.5 * (
            (sigma_w**2 + (self.mu_w - prior_mu_w) ** 2) / (sigma0**2)
            - 1.0
            + 2.0 * torch.log(sigma0 / sigma_w)
        ).sum()
        kl_b = 0.5 * (
            (sigma_b**2 + (self.mu_b - prior_mu_b) ** 2) / (sigma0**2)
            - 1.0
            + 2.0 * torch.log(sigma0 / sigma_b)
        ).sum()
        # Also regularize the gate towards "off" (s≈0) via a simple quadratic prior on alpha_logit.
        prior_a = self.prior_alpha_logit.to(self.alpha_logit.device)
        kl_a = 0.5 * ((self.alpha_logit - prior_a) ** 2) / (sigma0**2)
        return kl_w + kl_b + kl_a

    def kl_to_prev_posterior(self, prev: PosteriorParams) -> torch.Tensor:
        sigma0 = torch.tensor(self.sigma0, device=self.mu_w.device)
        sigma_w = torch.exp(self.log_sigma_w)
        sigma_b = torch.exp(self.log_sigma_b)
        prev_sigma_w = torch.exp(prev.log_sigma_w)
        prev_sigma_b = torch.exp(prev.log_sigma_b)
        kl_w = 0.5 * (
            (sigma_w**2 + (self.mu_w - prev.mu_w) ** 2) / (prev_sigma_w**2)
            - 1.0
            + 2.0 * torch.log(prev_sigma_w / sigma_w)
        ).sum()
        kl_b = 0.5 * (
            (sigma_b**2 + (self.mu_b - prev.mu_b) ** 2) / (prev_sigma_b**2)
            - 1.0
            + 2.0 * torch.log(prev_sigma_b / sigma_b)
        ).sum()
        # Trust-region also applies to the gate parameter.
        prev_a = prev.alpha_logit.to(self.alpha_logit.device)
        kl_a = 0.5 * ((self.alpha_logit - prev_a) ** 2) / (sigma0**2)
        return kl_w + kl_b + kl_a

    def snapshot(self) -> PosteriorParams:
        return PosteriorParams(
            mu_w=self.mu_w.detach().clone(),
            log_sigma_w=self.log_sigma_w.detach().clone(),
            mu_b=self.mu_b.detach().clone(),
            log_sigma_b=self.log_sigma_b.detach().clone(),
            alpha_logit=self.alpha_logit.detach().clone(),
        )
