from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from einops import rearrange
from transformers.models.gpt2.modeling_gpt2 import GPT2Model

from .base import Backbone

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(it, **kwargs):
        return it


class GPT4TSModel(nn.Module):
    """
    Port of OnlineTSF GPT4TS backbone for OMPB.

    Inputs:
      x: [B, L, C_in]
    Outputs:
      y: [B, pred_len, C_out]

    Note:
      GPT4TS's original formulation predicts per-channel; therefore we always feed only the first
      `out_channels` channels to the model (so appended time-features are ignored).
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        out_channels: int,
        patch_size: int,
        stride: int,
        d_model: int,
        gpt_layers: int,
        pretrained_name_or_path: str = "gpt2",
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.out_channels = int(out_channels)
        self.patch_size = int(patch_size)
        self.stride = int(stride)

        self.patch_num = (self.seq_len - self.patch_size) // self.stride + 1
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1

        self.gpt2 = self._load_gpt2(pretrained_name_or_path, d_model=int(d_model), gpt_layers=int(gpt_layers))

        self.in_layer = nn.Linear(self.patch_size, int(d_model))
        self.out_layer = nn.Linear(int(d_model) * self.patch_num, self.pred_len)

        # Freeze most GPT2 weights (paper-style fine-tuning: keep LN + positional embeddings).
        for name, param in self.gpt2.named_parameters():
            if "ln" in name or "wpe" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    @staticmethod
    def _choose_n_head(d_model: int) -> int:
        # Pick a reasonable n_head that divides d_model.
        for h in (16, 12, 8, 6, 4, 3, 2, 1):
            if d_model % h == 0:
                return h
        return 1

    @classmethod
    def _load_gpt2(cls, name_or_path: str, d_model: int, gpt_layers: int) -> GPT2Model:
        # Strict offline loading:
        # - Only load from local files (no network)
        # - NEVER fall back to random initialization
        try:
            m = GPT2Model.from_pretrained(
                name_or_path,
                local_files_only=True,
                output_attentions=False,
                output_hidden_states=False,
            )
            m.h = m.h[:gpt_layers]
            return m
        except Exception as e:
            raise RuntimeError(
                f"Failed to load pretrained GPT-2 from '{name_or_path}' with local_files_only=True. "
                f"Make sure the directory exists and contains GPT-2 files (e.g. config.json, model weights). "
                f"This OMPB GPT4TS backbone is configured to NEVER use random initialization."
            ) from e

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use only the real series channels; ignore any appended time features.
        x = x[:, :, : self.out_channels]
        b, l, c = x.shape
        if l != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {l}")

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = x / stdev

        # Patch along time, per channel
        x = rearrange(x, "b l c -> b c l")
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, "b c n p -> (b c) n p")

        h = self.in_layer(x)
        h = self.gpt2(inputs_embeds=h, output_attentions=False, output_hidden_states=False, use_cache=False).last_hidden_state

        out = self.out_layer(h.reshape(b * c, -1))
        out = rearrange(out, "(b c) l -> b l c", b=b)

        out = out * stdev + means
        return out


class GPT4TSBackbone(Backbone):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        out_channels: int,
        patch_size: int,
        stride: int,
        d_model: int,
        gpt_layers: int,
        lr: float,
        epochs: int,
        batch_size: int,
        device: str,
        pretrained_name_or_path: str = "gpt2",
    ) -> None:
        self.device = str(device)
        self.model = GPT4TSModel(
            seq_len=seq_len,
            pred_len=pred_len,
            out_channels=out_channels,
            patch_size=patch_size,
            stride=stride,
            d_model=d_model,
            gpt_layers=gpt_layers,
            pretrained_name_or_path=pretrained_name_or_path,
        ).to(self.device)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)

    def fit(self, train_dataset) -> None:
        loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        loss_fn = nn.MSELoss()
        self.model.train()
        for epoch in tqdm(range(self.epochs), desc="GPT4TS epochs"):
            pbar = tqdm(loader, desc=f"GPT4TS epoch {epoch+1}/{self.epochs}", leave=False)
            for x, y in pbar:
                x = x.to(self.device)
                y = y.to(self.device)
                pred = self.model(x)
                loss = loss_fn(pred, y)
                optim.zero_grad()
                loss.backward()
                optim.step()
                try:
                    pbar.set_postfix(loss=float(loss.detach().cpu().item()))
                except Exception:
                    pass

    def predict_batch(self, x_batch: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(x_batch.to(self.device))

