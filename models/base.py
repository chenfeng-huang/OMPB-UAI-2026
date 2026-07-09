from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Backbone(ABC):
    @abstractmethod
    def fit(self, train_dataset) -> None:
        pass

    @abstractmethod
    def predict_batch(self, x_batch: torch.Tensor) -> torch.Tensor:
        pass
