from typing import Dict, Optional

import lightning as pl
import numpy as np
import torch
from torch.utils.data import DataLoader

from config.dataloader_config import DataloaderConfig


class DataModuleHeatmap(pl.LightningDataModule):
    """Simple Lightning data module for heatmap datasets."""
    def __init__(self, datasets: Dict[str, torch.utils.data.Dataset], config: DataloaderConfig) -> None:
        """Store datasets and config."""
        super().__init__()
        self._config: DataloaderConfig = config
        self.sampler = None

        self.train_dataset = datasets["train"]
        self.val_dataset = datasets["val"]
        self.test_dataset = datasets["test"]

    def setup(self, stage: Optional[str] = None) -> None:
        """Prepare optional sampler before training."""
        if self._config.weighted_sampler and stage == "fit":
            self.sampler = self._build_weighted_sampler()
            print("Using weighted sampler for training data loader.")

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader."""
        if self.sampler is not None:
            return DataLoader(
                self.train_dataset,
                sampler=self.sampler,
                **self._config.train_dataloader.__dict__,
            )

        return DataLoader(self.train_dataset, **self._config.train_dataloader.__dict__)

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        return DataLoader(self.val_dataset, **self._config.validation_dataloader.__dict__)

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""
        return DataLoader(self.test_dataset, **self._config.test_dataloader.__dict__)

    def predict_dataloader(self) -> DataLoader:
        """Return the prediction dataloader."""
        return DataLoader(self.test_dataset, **self._config.test_dataloader.__dict__)

    def _build_weighted_sampler(self) -> torch.utils.data.WeightedRandomSampler:
        """Build a weighted sampler from target values."""
        bins = self._config.bins
        values = np.asarray([self.train_dataset[i][1] for i in range(len(self.train_dataset))]) # type: ignore
        edges = np.linspace(values.min(), values.max(), bins + 1)
        bin_ids = np.digitize(values, edges[1:-1], right=True)
        counts = np.bincount(bin_ids, minlength=bins)
        counts[counts == 0] = 1  # avoid division by zero
        weights = 1.0 / counts[bin_ids]
        return torch.utils.data.WeightedRandomSampler(
            weights, # type: ignore
            num_samples=len(self.train_dataset), # type: ignore
            replacement=True,
        )