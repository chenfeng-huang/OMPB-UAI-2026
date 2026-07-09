from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


@dataclass
class ETTConfig:
    etth1_path: str
    etth2_path: str
    split: Dict[str, float]
    seq_len: int
    pred_len: int
    scaler_path: str
    # How to handle feature-set mismatches between etth1/etth2 CSVs.
    # - "strict": require identical columns (default, original behavior)
    # - "train": keep train(etth1) columns; reindex etth2 and fill missing with NaN
    # - "intersection": keep only common columns (in etth1 column order)
    align_columns: str = "strict"
    # If true, impute NaNs using train-set column means before scaling/windowing.
    impute_nan: bool = False


class ETTWindowDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int) -> None:
        self.data = data.astype(np.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return torch.from_numpy(x), torch.from_numpy(y)


def _load_csv(path: str) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = df[numeric_cols]
    return df, numeric_cols


def _align_feature_columns(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Align feature columns between train and test frames.
    """
    mode = str(mode or "strict").strip().lower()
    cols_train = list(df_train.columns)
    cols_test = list(df_test.columns)

    if mode == "strict":
        if cols_train != cols_test:
            raise ValueError("ETTh1 and ETTh2 columns do not match.")
        return df_train, df_test, cols_train

    if mode == "intersection":
        keep = [c for c in cols_train if c in set(cols_test)]
        return df_train[keep], df_test[keep], keep

    if mode == "train":
        # Keep train columns; add any missing columns to test as NaNs and drop extras.
        df_test_aligned = df_test.reindex(columns=cols_train)
        return df_train[cols_train], df_test_aligned, cols_train

    raise ValueError(f"Unknown align_columns mode: {mode}. Use strict|train|intersection")


def _impute_nan_with_train_means(train: np.ndarray, other: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Impute NaNs using per-feature means computed on train (ignoring NaNs).
    Any feature with all-NaN mean is imputed with 0.0.
    """
    means = np.nanmean(train, axis=0)
    means = np.where(np.isfinite(means), means, 0.0).astype(np.float32)
    train_out = np.array(train, copy=True)
    other_out = np.array(other, copy=True)
    train_nan = ~np.isfinite(train_out)
    other_nan = ~np.isfinite(other_out)
    if train_nan.any():
        train_out[train_nan] = np.take(means, np.where(train_nan)[1])
    if other_nan.any():
        other_out[other_nan] = np.take(means, np.where(other_nan)[1])
    return train_out.astype(np.float32), other_out.astype(np.float32)


def _split_indices(n: int, split: Dict[str, float]) -> Dict[str, Tuple[int, int]]:
    train_end = int(n * split["train"])
    val_end = train_end + int(n * split["val"])
    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, n),
    }


def _scale_data(
    train_data: np.ndarray, full_data: np.ndarray, scaler_path: str
) -> Tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    scaler.fit(train_data)
    scaled = scaler.transform(full_data)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    return scaled, scaler


def load_ett_datasets(
    config: ETTConfig,
) -> Dict[str, Dict[str, ETTWindowDataset]]:
    df1, cols1 = _load_csv(config.etth1_path)
    df2, cols2 = _load_csv(config.etth2_path)
    df1, df2, cols = _align_feature_columns(df1, df2, mode=getattr(config, "align_columns", "strict"))

    split_indices = _split_indices(len(df1), config.split)

    train_slice = split_indices["train"]
    train_data = df1.iloc[train_slice[0] : train_slice[1]].values.astype(np.float32)
    full1 = df1.values.astype(np.float32)
    full2 = df2.values.astype(np.float32)

    if bool(getattr(config, "impute_nan", False)):
        train_data, full1 = _impute_nan_with_train_means(train_data, full1)
        _, full2 = _impute_nan_with_train_means(train_data, full2)

    scaled1, _ = _scale_data(train_data, full1, config.scaler_path)
    scaled2 = joblib.load(config.scaler_path).transform(full2)

    datasets = {}
    for name, (start, end) in split_indices.items():
        datasets.setdefault("etth1", {})[name] = ETTWindowDataset(
            scaled1[start:end],
            seq_len=config.seq_len,
            pred_len=config.pred_len,
        )
    datasets.setdefault("etth2", {})["test"] = ETTWindowDataset(
        scaled2,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
    )
    datasets["meta"] = {
        "feature_names": cols,
        "x_channels": int(len(cols)),
        "y_channels": int(len(cols)),
    }
    return datasets


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
