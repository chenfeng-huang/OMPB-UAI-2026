from __future__ import annotations

from typing import List

import torch
from torch import nn
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader

from .base import Backbone

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(it, **kwargs):
        return it


# TCN (Bai, Kolter, Koltun 2018):
# - Causal 1D convolutions via padding + chomp
# - Dilations grow exponentially
# - Residual blocks with weight normalization, dropout, ReLU


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size <= 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.drop1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.drop2,
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

        self._init_weights()

    def _init_weights(self) -> None:
        # Common TCN init: normal(0, 0.01) for conv weights
        for m in [self.conv1, self.conv2]:
            nn.init.normal_(m.weight, 0.0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, 0.0, 0.01)
            if self.downsample.bias is not None:
                nn.init.zeros_(self.downsample.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs: int, num_channels: List[int], kernel_size: int, dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        for i, out_ch in enumerate(num_channels):
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            dilation = 2**i
            layers.append(
                TemporalBlock(
                    n_inputs=in_ch,
                    n_outputs=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TCNModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        widths: List[int],
        kernel_size: int,
        dropout: float,
        pred_len: int,
        out_channels: int = 1,
    ):
        super().__init__()
        self.tcn = TemporalConvNet(in_channels, widths, kernel_size=kernel_size, dropout=dropout)
        self.pred_len = int(pred_len)
        self.out_channels = int(out_channels)
        self.head = nn.Linear(widths[-1], self.pred_len * self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, features]
        x = x.transpose(1, 2)  # -> [batch, features, seq_len]
        out = self.tcn(x)
        last = out[:, :, -1]
        y = self.head(last).view(x.shape[0], self.pred_len, self.out_channels)
        # Backward-compatible: when predicting a single target, return [B, pred_len]
        return y.squeeze(-1) if self.out_channels == 1 else y


class TCNBackbone(Backbone):
    def __init__(
        self,
        in_channels: int,
        pred_len: int,
        widths: List[int],
        kernel_size: int,
        dropout: float,
        lr: float,
        epochs: int,
        batch_size: int,
        device: str,
        out_channels: int = 1,
    ) -> None:
        self.device = device
        self.model = TCNModel(in_channels, widths, kernel_size, dropout, pred_len, out_channels=out_channels).to(device)
        # YAML can parse values like "1e-3" as strings; make this robust.
        self.lr = float(lr)
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, train_dataset) -> None:
        loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        self.model.train()
        for epoch in tqdm(range(self.epochs), desc="TCN epochs"):
            pbar = tqdm(loader, desc=f"TCN epoch {epoch+1}/{self.epochs}", leave=False)
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
