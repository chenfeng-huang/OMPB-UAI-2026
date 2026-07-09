from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


@dataclass
class ILIConfig:
    train_path: str
    test_path: str
    split: Dict[str, float]
    seq_len: int
    pred_len: int
    scaler_path: str
    # Align test columns to train columns (recommended).
    align_columns: str = "train"  # train | strict | intersection
    # ILI often contains missing values; StandardScaler cannot handle NaNs.
    impute_nan: bool = True


class ILIWindowDataset(Dataset):
    """
    Simple sliding-window dataset for ILI numeric arrays.

    Shapes:
      - data: [T, F] float32
      - __getitem__ returns:
          x: [seq_len, F]
          y: [pred_len, F]
    """

    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int) -> None:
        self.data = data.astype(np.float32)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)

    def __len__(self) -> int:
        return max(0, int(len(self.data) - self.seq_len - self.pred_len + 1))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return torch.from_numpy(x), torch.from_numpy(y)


def _split_indices(n: int, split: Dict[str, float]) -> Dict[str, Tuple[int, int]]:
    train_end = int(n * float(split["train"]))
    val_end = train_end + int(n * float(split["val"]))
    return {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, n)}


def _align_feature_columns(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    mode = str(mode or "train").strip().lower()
    cols_train = list(df_train.columns)
    cols_test = list(df_test.columns)

    if mode == "strict":
        if cols_train != cols_test:
            raise ValueError("ILI train/test columns do not match (strict).")
        return df_train, df_test, cols_train

    if mode == "intersection":
        keep = [c for c in cols_train if c in set(cols_test)]
        return df_train[keep], df_test[keep], keep

    if mode == "train":
        df_test_aligned = df_test.reindex(columns=cols_train)
        return df_train[cols_train], df_test_aligned, cols_train

    raise ValueError(f"Unknown align_columns mode: {mode}. Use train|strict|intersection")


def _impute_nan_with_train_means(train_slice: np.ndarray, other: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    means = np.nanmean(train_slice, axis=0)
    means = np.where(np.isfinite(means), means, 0.0).astype(np.float32)

    train_out = np.array(train_slice, copy=True)
    other_out = np.array(other, copy=True)

    train_nan = ~np.isfinite(train_out)
    other_nan = ~np.isfinite(other_out)
    if train_nan.any():
        train_out[train_nan] = np.take(means, np.where(train_nan)[1])
    if other_nan.any():
        other_out[other_nan] = np.take(means, np.where(other_nan)[1])
    return train_out.astype(np.float32), other_out.astype(np.float32)


def _scale_data(
    train_slice: np.ndarray, full: np.ndarray, scaler_path: str
) -> Tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    scaler.fit(train_slice)
    scaled = scaler.transform(full)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    return scaled.astype(np.float32), scaler


def load_ili_datasets(config: ILIConfig) -> Dict[str, Dict[str, ILIWindowDataset]]:
    df_train = pd.read_csv(config.train_path)
    df_test = pd.read_csv(config.test_path)

    # Ensure numeric-only ordering is consistent.
    num_train = df_train.select_dtypes(include=[np.number])
    num_test = df_test.select_dtypes(include=[np.number])
    num_train, num_test, cols = _align_feature_columns(num_train, num_test, mode=config.align_columns)

    split_idx = _split_indices(len(num_train), config.split)
    train_start, train_end = split_idx["train"]

    train_slice = num_train.iloc[train_start:train_end].values.astype(np.float32)
    full_train = num_train.values.astype(np.float32)
    full_test = num_test.values.astype(np.float32)

    if bool(config.impute_nan):
        train_slice, full_train = _impute_nan_with_train_means(train_slice, full_train)
        _, full_test = _impute_nan_with_train_means(train_slice, full_test)

    scaled_train, _ = _scale_data(train_slice, full_train, config.scaler_path)
    scaled_test = joblib.load(config.scaler_path).transform(full_test).astype(np.float32)

    out: Dict[str, Dict[str, ILIWindowDataset]] = {"ili_train": {}, "ili_test": {}}
    for name, (start, end) in split_idx.items():
        out["ili_train"][name] = ILIWindowDataset(scaled_train[start:end], seq_len=config.seq_len, pred_len=config.pred_len)
    out["ili_test"]["test"] = ILIWindowDataset(scaled_test, seq_len=config.seq_len, pred_len=config.pred_len)
    out["meta"] = {
        "feature_names": cols,
        "x_channels": int(len(cols)),
        "y_channels": int(len(cols)),
    }
    return out

