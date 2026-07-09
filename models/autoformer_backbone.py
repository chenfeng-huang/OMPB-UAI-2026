from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from .base import Backbone

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(it, **kwargs):
        return it


class MovingAverage(nn.Module):
    """
    Moving average used in Autoformer series decomposition.
    Implements same-length smoothing via padding.
    """

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.avg = nn.AvgPool1d(self.kernel_size, stride=1, padding=self.kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        x_t = x.transpose(1, 2)  # [B, C, L]
        ma = self.avg(x_t)
        return ma.transpose(1, 2)  # [B, L, C]


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = MovingAverage(kernel_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class AutoCorrelation(nn.Module):
    """
    Auto-Correlation mechanism (Wu et al., 2021) simplified to work without time features.
    Uses FFT-based correlation + time-delay aggregation.
    """

    def __init__(self, factor: float = 3.0, dropout: float = 0.0):
        super().__init__()
        self.factor = float(factor)
        self.dropout = nn.Dropout(dropout)

    def _time_delay_agg(self, values: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        """
        values: [B, H, L, D]
        corr:   [B, H, L] correlation scores per delay (only depends on delay)
        """
        b, h, l, d = values.shape
        # Top-k delays (paper: k = factor * log(L))
        k = max(1, int(self.factor * float(torch.log(torch.tensor(float(l))).item())))
        topk = torch.topk(corr, k=k, dim=-1)  # (vals, idx) each [B, H, k]
        weights = torch.softmax(topk.values, dim=-1)  # [B, H, k]
        delays = topk.indices  # [B, H, k]

        # Vectorized per-(B,H) rolling via gather (avoids Python loops that peg CPU).
        # For a left roll by `delay`: out[t] = values[(t + delay) % L]
        t = torch.arange(l, device=values.device, dtype=torch.long)  # [L]
        idx = (t.view(1, 1, 1, l) + delays.to(torch.long).unsqueeze(-1)) % l  # [B,H,k,L]
        idx = idx.unsqueeze(-1).expand(b, h, k, l, d)  # [B,H,k,L,D]
        v = values.unsqueeze(2).expand(b, h, k, l, d)  # [B,H,k,L,D]
        shifted = v.gather(3, idx)  # [B,H,k,L,D]
        out = (weights.unsqueeze(-1).unsqueeze(-1) * shifted).sum(dim=2)  # [B,H,L,D]
        return self.dropout(out)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        q: [B, H, Lq, D], k: [B, H, Lk, D], v: [B, H, Lv, D]
        returns: [B, H, Lq, D]
        """
        lq = int(q.shape[2])
        lk = int(k.shape[2])
        lv = int(v.shape[2])
        n = max(lq, lk, lv)

        def _pad_to(x: torch.Tensor, n_: int) -> torch.Tensor:
            if int(x.shape[2]) == n_:
                return x
            pad_len = n_ - int(x.shape[2])
            pad = torch.zeros(x.shape[0], x.shape[1], pad_len, x.shape[3], device=x.device, dtype=x.dtype)
            return torch.cat([x, pad], dim=2)

        q_pad = _pad_to(q, n)
        k_pad = _pad_to(k, n)
        v_pad = _pad_to(v, n)

        # FFT-based cross-correlation along time dimension (length n).
        qf = torch.fft.rfft(q_pad, n=n, dim=2)
        kf = torch.fft.rfft(k_pad, n=n, dim=2)
        corr = torch.fft.irfft(qf * torch.conj(kf), n=n, dim=2)  # [B,H,n,D]
        corr = corr.mean(dim=-1)  # [B,H,n]
        out = self._time_delay_agg(v_pad, corr)  # [B,H,n,D]
        return out[:, :, :lq, :]


class AutoCorrelationLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, factor: float, dropout: float):
        super().__init__()
        self.n_heads = int(n_heads)
        self.d_head = d_model // self.n_heads
        if d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads for Autoformer")

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.ac = AutoCorrelation(factor=factor, dropout=dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,L,d_model] -> [B,H,L,d_head]
        b, l, _ = x.shape
        x = x.view(b, l, self.n_heads, self.d_head).transpose(1, 2)
        return x

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,H,L,d_head] -> [B,L,d_model]
        b, h, l, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, l, h * d)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        q = self._split_heads(self.q_proj(x_q))
        k = self._split_heads(self.k_proj(x_kv))
        v = self._split_heads(self.v_proj(x_kv))
        out = self.ac(q, k, v)
        return self.out_proj(self._merge_heads(out))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, factor: float, moving_avg: int, dropout: float):
        super().__init__()
        self.attn = AutoCorrelationLayer(d_model, n_heads, factor=factor, dropout=dropout)
        self.decomp1 = SeriesDecomp(moving_avg)
        self.ff = FeedForward(d_model, d_ff, dropout=dropout)
        self.decomp2 = SeriesDecomp(moving_avg)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # returns (seasonal, trend)
        x = x + self.attn(x, x)
        seasonal, trend1 = self.decomp1(x)
        seasonal = seasonal + self.ff(seasonal)
        seasonal, trend2 = self.decomp2(seasonal)
        return seasonal, trend1 + trend2


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, factor: float, moving_avg: int, dropout: float):
        super().__init__()
        self.self_attn = AutoCorrelationLayer(d_model, n_heads, factor=factor, dropout=dropout)
        self.cross_attn = AutoCorrelationLayer(d_model, n_heads, factor=factor, dropout=dropout)
        self.decomp1 = SeriesDecomp(moving_avg)
        self.decomp2 = SeriesDecomp(moving_avg)
        self.decomp3 = SeriesDecomp(moving_avg)
        self.ff = FeedForward(d_model, d_ff, dropout=dropout)
        self.trend_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, enc: torch.Tensor, trend: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x + self.self_attn(x, x)
        seasonal, trend1 = self.decomp1(x)
        seasonal = seasonal + self.cross_attn(seasonal, enc)
        seasonal, trend2 = self.decomp2(seasonal)
        seasonal = seasonal + self.ff(seasonal)
        seasonal, trend3 = self.decomp3(seasonal)
        trend = trend + self.trend_proj(trend1 + trend2 + trend3)
        return seasonal, trend


class AutoformerModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        pred_len: int,
        d_model: int,
        n_heads: int,
        e_layers: int,
        d_layers: int,
        moving_avg: int,
        dropout: float,
        out_channels: int = 1,
        factor: float = 3.0,
        d_ff: Optional[int] = None,
        label_len: Optional[int] = None,
    ):
        super().__init__()
        self.n_features = int(n_features)
        self.pred_len = int(pred_len)
        self.d_model = int(d_model)
        self.out_channels = int(out_channels)
        self.label_len = label_len

        d_ff = int(d_ff) if d_ff is not None else 4 * self.d_model

        self.decomp = SeriesDecomp(moving_avg)
        self.enc_embed = nn.Linear(self.n_features, self.d_model)
        self.dec_embed = nn.Linear(self.n_features, self.d_model)

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=self.d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    factor=factor,
                    moving_avg=moving_avg,
                    dropout=dropout,
                )
                for _ in range(int(e_layers))
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=self.d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    factor=factor,
                    moving_avg=moving_avg,
                    dropout=dropout,
                )
                for _ in range(int(d_layers))
            ]
        )

        # Final projections: seasonal+trend -> forecast
        self.proj_seasonal = nn.Linear(self.d_model, self.out_channels)
        self.proj_trend = nn.Linear(self.d_model, self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        b, l, c = x.shape
        label_len = self.label_len if self.label_len is not None else max(1, l // 2)

        seasonal_init, trend_init = self.decomp(x)
        # Decoder inputs (paper-style): seasonal = [last label_len seasonal, zeros pred_len]
        # trend = [last label_len trend, repeat mean pred_len]
        mean = x.mean(dim=1, keepdim=True)
        zeros = torch.zeros(b, self.pred_len, c, device=x.device, dtype=x.dtype)
        seasonal_dec = torch.cat([seasonal_init[:, -label_len:, :], zeros], dim=1)
        trend_dec = torch.cat([trend_init[:, -label_len:, :], mean.repeat(1, self.pred_len, 1)], dim=1)

        enc = self.enc_embed(x)
        trend_enc_total = torch.zeros_like(enc)
        for layer in self.encoder_layers:
            enc, trend_delta = layer(enc)
            trend_enc_total = trend_enc_total + trend_delta

        dec = self.dec_embed(seasonal_dec)
        trend = self.dec_embed(trend_dec)
        for layer in self.decoder_layers:
            dec, trend = layer(dec, enc, trend)

        out = self.proj_seasonal(dec) + self.proj_trend(trend)  # [B, L_dec, out_channels]
        out = out[:, -self.pred_len :, :]  # [B, pred_len, out_channels]
        return out[:, :, 0] if self.out_channels == 1 else out


class AutoformerBackbone(Backbone):
    def __init__(
        self,
        n_features: int,
        pred_len: int,
        d_model: int,
        n_heads: int,
        e_layers: int,
        moving_avg: int,
        dropout: float,
        lr: float,
        epochs: int,
        batch_size: int,
        device: str,
        out_channels: int = 1,
        d_layers: int = 1,
        factor: float = 3.0,
        d_ff: Optional[int] = None,
        label_len: Optional[int] = None,
    ) -> None:
        self.device = device
        self.model = AutoformerModel(
            n_features=n_features,
            pred_len=pred_len,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=d_layers,
            moving_avg=moving_avg,
            dropout=dropout,
            out_channels=out_channels,
            factor=factor,
            d_ff=d_ff,
            label_len=label_len,
        ).to(device)
        # YAML can parse values like "1e-3" as strings; make this robust.
        self.lr = float(lr)
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, train_dataset) -> None:
        loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        self.model.train()
        for epoch in tqdm(range(self.epochs), desc="Autoformer epochs"):
            pbar = tqdm(loader, desc=f"Autoformer epoch {epoch+1}/{self.epochs}", leave=False)
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
            preds = self.model(x_batch.to(self.device))
        # Keep predictions on the backbone's device; callers can move to CPU if needed.
        # Returning CPU here causes costly cuda->cpu->cuda hops in downstream OMPB code.
        return preds
