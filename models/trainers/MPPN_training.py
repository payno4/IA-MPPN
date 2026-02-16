"""MPPN model training orchestration using PyTorch Lightning.

This module provides the MPPNTrainer class which manages the complete training lifecycle
for Multi-scale Point-wise Pattern Networks (MPPN) including training, testing, prediction,
and evaluation stages with MLFlow experiment tracking and checkpoint management.
"""

from pathlib import Path

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from lightning import LightningDataModule

from config.configs import Configs
from models.MPPN.lightning_module_mppn import MPPNLightningModule


class MPPNTrainer:
    """Orchestrates MPPN model training and evaluation using PyTorch Lightning.

    This trainer manages the complete training pipeline including checkpoint saving,
    MLFlow experiment tracking, testing, prediction, and evaluation stages.
    """

    def __init__(
        self,
        config: Configs,
        dataset: LightningDataModule,
        filename_chpt: str = "bestmodel_MPPN_PermitLog_{epoch:02d}-{valloss:02d}",
        monitor1: str = "val/metrics/R2",
        monitor1_mode: str = "max",
        prefix: str = "DomDec_",
    ) -> None:
        """Initialize the MPPN trainer with configuration and dataset.

        Args:
            config: Configuration object containing model, training, and IO settings.
            dataset: PyTorch Lightning DataModule providing train/val/test splits.
            filename_chpt: Template for checkpoint filename. Defaults to
                "bestmodel_MPPN_PermitLog_{epoch:02d}-{valloss:02d}".
            monitor1: Metric to monitor for checkpoint saving. Defaults to "val/metrics/R2".
            monitor1_mode: Direction to optimize monitored metric ("max" or "min").
                Defaults to "max".
            prefix: Prefix for checkpoint and MLFlow database filenames. Defaults to "DomDec_".
        """
        self.config = config
        self.dataset = dataset

        mlf_logger = self._setup_mlflow_logger(prefix)
        checkpoint_callback = self._setup_checkpoint_callback(
            filename_chpt, monitor1, monitor1_mode, prefix
        )

        self.trainer = self._setup_trainer(mlf_logger, checkpoint_callback)
        self.model = MPPNLightningModule(config)

    def _setup_mlflow_logger(self, prefix: str) -> MLFlowLogger:
        """Create and configure MLFlow logger for experiment tracking.

        Args:
            prefix: Prefix for MLFlow database filename.

        Returns:
            Configured MLFlowLogger instance.
        """
        mlflow_dir: str = self.config.io_config.mlflow_dir.as_posix()
        return MLFlowLogger(
            experiment_name=self.config.experiment_config.experiment_name,
            run_name=self.config.experiment_config.run_name,
            artifact_location=mlflow_dir,
            tracking_uri=f"sqlite:///{mlflow_dir}/{prefix}mlflow_MPPN.db",
        )

    def _setup_checkpoint_callback(
        self, filename_chpt: str, monitor1: str, monitor1_mode: str, prefix: str
    ) -> ModelCheckpoint:
        """Create checkpoint callback for saving best model during training.

        Args:
            filename_chpt: Template for checkpoint filename.
            monitor1: Metric to monitor for checkpoint saving.
            monitor1_mode: Direction to optimize metric ("max" or "min").
            prefix: Prefix to add to checkpoint filename.

        Returns:
            Configured ModelCheckpoint callback.
        """
        filename_chpt = prefix + filename_chpt
        return ModelCheckpoint(
            filename=filename_chpt,
            monitor=monitor1,
            dirpath=self.config.io_config.checkpoints_dir,
            mode=monitor1_mode,
        )

    def _setup_trainer(
        self, mlf_logger: MLFlowLogger, checkpoint_callback: ModelCheckpoint
    ) -> pl.Trainer:
        """Create and configure PyTorch Lightning Trainer instance.

        Args:
            mlf_logger: MLFlow logger for experiment tracking.
            checkpoint_callback: Checkpoint callback for model saving.

        Returns:
            Configured Trainer instance.
        """
        return pl.Trainer(
            deterministic=True,
            accelerator=self.config.experiment_config.device,
            devices="auto",
            logger=mlf_logger,
            max_epochs=self.config.trainer_config.epochs,
            log_every_n_steps=10,
            check_val_every_n_epoch=1,
            fast_dev_run=False,
            gradient_clip_val=None,
            num_sanity_val_steps=1,
            precision="16-mixed",
            profiler="simple",
            enable_progress_bar=True,
            use_distributed_sampler=False,
            val_check_interval=1.0,
            enable_model_summary=False,
            callbacks=[checkpoint_callback],
        )

    def train(self, ckpt_path: str | None = None) -> None:
        """Train the MPPN model on the training dataset.

        Performs model training for the configured number of epochs with validation
        checks and saves the final model after training completes.

        Args:
            ckpt_path: Path to checkpoint to resume training from. If None, trains from scratch.
        """
        print(f"Starting MPPN training with {self.trainer.max_epochs} epochs.")
        self.trainer.fit(
            self.model, datamodule=self.dataset, ckpt_path=ckpt_path
        )  # type: ignore
        final_ckpt_path: Path = self.config.io_config.checkpoints_dir / "final_model_MPPN.ckpt"
        self.trainer.save_checkpoint(final_ckpt_path)

    def test(self, ckpt_path: str | None = None) -> None:
        """Evaluate the MPPN model on the test dataset.

        Runs the model in evaluation mode on the test split and logs metrics.

        Args:
            ckpt_path: Path to checkpoint to load for testing. If None, uses current model.
        """
        print("Starting MPPN testing.")
        self.trainer.test(
            self.model, datamodule=self.dataset, ckpt_path=ckpt_path
        )  # type: ignore

    def predict(self, ckpt_path: str | None = None) -> None:
        """Generate predictions on the dataset using the trained MPPN model.

        Runs inference mode on the dataset and returns predictions without computing loss.

        Args:
            ckpt_path: Path to checkpoint to load for prediction. If None, uses current model.
        """
        print("Starting MPPN prediction.")
        self.trainer.predict(
            self.model, datamodule=self.dataset, ckpt_path=ckpt_path
        )

    def evaluate(self, ckpt_path: str | None = None) -> None:
        """Evaluate model performance on test dataset.

        Wrapper around test() method for explicit evaluation interface.

        Args:
            ckpt_path: Path to checkpoint to load for evaluation. If None, uses current model.
        """
        self.trainer.test(
            self.model, datamodule=self.dataset, ckpt_path=ckpt_path
        )

