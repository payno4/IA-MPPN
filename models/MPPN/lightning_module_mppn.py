"""Lightning module for MPPN model training and evaluation."""

from typing import Any, Dict, List, Tuple

import lightning as pl
import torch
import torchmetrics
from lightning.pytorch.loggers import MLFlowLogger
from mlflow.tracking.client import MlflowClient
from torch import nn

from config.configs import Configs
from config.model_configs.MPPN_configs import MPPNConfig
from models.IA_ViT.losses import MMDLoss
from models.MPPN.original_MPPN import MPPNRegressor
from pipeline.utils import get_config_details

class MPPNLightningModule(pl.LightningModule):
    """Lightning module for MPPN model training and evaluation."""

    def __init__(
        self, config: Configs, model: MPPNRegressor | None = None, learning_rate: float | None = None
    ) -> None:
        """Initialize the lightning module.
        
        Args:
            config: Configuration object.
            model: Optional pre-built MPPNRegressor model. If None, creates new model.
            learning_rate: Optional learning rate override.
        """
        assert isinstance(
            config.model_config, MPPNConfig
        ), "config.model_config must be an instance of MPPNConfig for MPPN model."
        super().__init__()
        self.config = config
        self.learning_rate = learning_rate or config.trainer_config.lr

        # Initialize model
        num_perspectives = config.dataloader_config.input_shape[2]
        self.model = model or MPPNRegressor(
            config=config.model_config, num_perspectives=num_perspectives
        )

        # Storage for outputs
        self.val_outs: List[Dict[str, torch.Tensor]] = []
        self.test_outs: List[Dict[str, torch.Tensor]] = []
        self.train_outs: List[Dict[str, torch.Tensor]] = []

        # Loss function
        self.criterion = self._build_criterion()
        
        # Metrics
        self.metrics = ["MAE", "MSE", "R2"]
        self._build_metrics()

        # MMD loss
        self.mmd_loss_fn = MMDLoss(sigma=1.0, kernel="rbf")

    def _build_criterion(self) -> nn.Module:
        """Build loss criterion based on configuration."""
        criterion_name = self.config.trainer_config.criterion
        if criterion_name == "MSELoss":
            return nn.MSELoss()
        elif criterion_name == "L1Loss":
            return nn.L1Loss()
        elif criterion_name == "SmoothL1Loss":
            return nn.SmoothL1Loss(beta=self.config.trainer_config.beta1)
        else:
            raise ValueError(f"Unknown criterion: {criterion_name}")

    def _build_metrics(self) -> None:
        """Build validation and test metrics."""
        metric_classes = {
            "MAE": torchmetrics.MeanAbsoluteError,
            "MSE": torchmetrics.MeanSquaredError,
            "R2": torchmetrics.R2Score,
        }
        for metric_name, metric_class in metric_classes.items():
            if metric_name in self.metrics:
                for data_type in ["val", "test"]:
                    setattr(self, f"{data_type}/metrics/{metric_name}", metric_class())

    def on_fit_start(self) -> None:
        """Log hyperparameters and experiment tags at training start."""
        if not isinstance(self.logger, MLFlowLogger):
            return
            
        self.logger.log_hyperparams(get_config_details(self.config))
        experiment = self.logger.experiment
        
        # Only add MPPN-specific tags if using MPPNConfig
        if not isinstance(self.config.model_config, MPPNConfig):
            return
            
        tags = {
            "Model": self.config.experiment_config.model_name,
            "Dataset": str(self.config.dataloader_config.h5_file_name),
            "feature_size": str(self.config.model_config.feature_size),  # type: ignore
            "representation_dim": str(self.config.model_config.representation_dim),  # type: ignore
        }
        for key, value in tags.items():
            experiment.set_tag(self.logger.run_id, key, value)

    def on_train_start(self) -> None:
        """Log learning rate at training start."""
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        print(f"Used lr: {lr}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor.
            
        Returns:
            Model predictions.
        """
        return self.model(x)
    
    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, Any], batch_idx: int
    ) -> torch.Tensor:
        """Training step.
        
        Args:
            batch: Tuple of (inputs, targets, _).
            batch_idx: Batch index.
            
        Returns:
            Training loss.
        """
        inputs, targets, _ = batch
        y_hat = self(inputs)
        loss = self.criterion(y_hat, targets.float().view_as(y_hat))
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        """Log learning rate at end of training epoch."""
        if not isinstance(self.logger, MLFlowLogger):
            return
        learning_rate = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.logger.experiment.log_metric(
            self.logger.run_id,
            "train/learning_rate",
            learning_rate,
            step=self.current_epoch,
        )
    def _compute_metrics(
        self, logits: torch.Tensor, targets: torch.Tensor, data_type: str
    ) -> None:
        """Compute and log metrics for validation/test.
        
        Args:
            logits: Model predictions.
            targets: Ground truth targets.
            data_type: Either "val" or "test".
        """
        mask = (~torch.isnan(targets)) & (~torch.isnan(logits))
        logits_valid = logits[mask]
        targets_valid = targets[mask]

        if logits_valid.numel() <= 1 or targets_valid.numel() <= 1:
            return

        for metric_name in self.metrics:
            if metric_name == "R2":
                # Denormalize for R2 score
                min_t = torch.tensor(
                    self.config.dataloader_config.target_min,
                    device=logits_valid.device,
                    dtype=torch.float32,
                )
                max_t = torch.tensor(
                    self.config.dataloader_config.target_max,
                    device=logits_valid.device,
                    dtype=torch.float32,
                )
                logits_r2 = logits_valid.float() * (max_t - min_t) + min_t
                targets_r2 = targets_valid.float() * (max_t - min_t) + min_t
                getattr(self, f"{data_type}/metrics/{metric_name}")(logits_r2, targets_r2)
            else:
                getattr(self, f"{data_type}/metrics/{metric_name}")(logits_valid, targets_valid)

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, Any], batch_idx: int
    ) -> torch.Tensor:
        """Validation step.
        
        Args:
            batch: Tuple of (inputs, targets, _).
            batch_idx: Batch index.
            
        Returns:
            Validation loss.
        """
        inputs, targets, _ = batch
        y_hat = self(inputs)
        targets = targets.float().view_as(y_hat)
        loss = self.criterion(y_hat, targets)
        
        self._compute_metrics(y_hat, targets, "val")
        self.val_outs.append({
            "val_loss": loss.detach().float().cpu(),
            "predictions": y_hat.detach().float().cpu(),
            "targets": targets.detach().float().cpu(),
            "inputs": inputs.detach().float().cpu(),
        })
        return loss
    def _aggregate_validation_outputs(
        self,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], int, int]:
        """Aggregate validation outputs across batches.
        
        Returns:
            Tuple of (predictions, targets, losses, total_samples, nan_samples).
        """
        all_predictions = []
        all_targets = []
        all_losses = []
        total_samples = 0
        nan_samples = 0

        for output in self.val_outs:
            val_loss = output["val_loss"]
            predictions = output["predictions"]
            targets = output["targets"]
            total_samples += len(predictions)
            all_losses.append(val_loss)

            valid_mask = ~torch.isnan(targets)
            nan_samples += int((~valid_mask).sum().item())

            if valid_mask.any():
                all_predictions.append(predictions[valid_mask])
                all_targets.append(targets[valid_mask])

        return all_predictions, all_targets, all_losses, total_samples, nan_samples

    def on_validation_epoch_end(self) -> None:
        """Log validation metrics at end of epoch."""
        all_predictions, all_targets, all_losses, total_samples, nan_samples = (
            self._aggregate_validation_outputs()
        )
        
        if not isinstance(self.logger, MLFlowLogger):
            self.val_outs.clear()
            return
            
        experiment = self.logger.experiment

        # Log epoch loss
        if len(all_losses) > 0:
            self.log("val/epoch_loss", torch.stack(all_losses).mean(), prog_bar=True)

        # Log NaN statistics
        if total_samples > 0:
            nan_percentage = nan_samples / total_samples * 100.0
            experiment.log_metric(
                self.logger.run_id,
                "val/nan_percentage",
                nan_percentage,
                step=self.current_epoch,
            )
            experiment.log_metric(
                self.logger.run_id,
                "val/epoch_nan_count",
                value=nan_samples,
                step=self.current_epoch,
            )

        # Log prediction/target statistics
        if len(all_predictions) > 0:
            predictions = torch.cat(all_predictions)
            targets = torch.cat(all_targets)
            stats = {
                "val/pred_var": torch.var(predictions),
                "val/pred_min": torch.min(predictions),
                "val/pred_max": torch.max(predictions),
                "val/target_var": torch.var(targets),
                "val/target_min": torch.min(targets),
                "val/target_max": torch.max(targets),
            }
            for key, value in stats.items():
                experiment.log_metric(self.logger.run_id, key, value, step=self.current_epoch)

        self.val_outs.clear()

        # Log metrics
        for metric in self.metrics:
            metric_fn = getattr(self, f"val/metrics/{metric}")
            try:
                metric_value = metric_fn.compute()
                self.log(f"val/metrics/{metric}", metric_value, prog_bar=True)
            except Exception:
                pass
            finally:
                metric_fn.reset()


    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build optimizer based on configuration.
        
        Returns:
            Configured optimizer.
        """
        optimizer_type = self.config.trainer_config.optimizer
        weight_decay = self.config.trainer_config.weight_decay

        if optimizer_type == "Adam":
            return torch.optim.Adam(
                self.parameters(), lr=self.learning_rate, weight_decay=weight_decay
            )
        elif optimizer_type == "AdamW":
            return torch.optim.AdamW(
                self.parameters(), lr=self.learning_rate, weight_decay=weight_decay
            )
        elif optimizer_type == "SGD":
            return torch.optim.SGD(
                self.parameters(), lr=self.learning_rate, weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    def _build_scheduler(self, optimizer: torch.optim.Optimizer) -> Dict[str, Any] | None:
        """Build learning rate scheduler based on configuration.
        
        Args:
            optimizer: Optimizer instance.
            
        Returns:
            Scheduler configuration dict or None.
        """
        scheduler_type = self.config.trainer_config.scheduler
        warmup_ratio = self.config.trainer_config.warmup_ratio

        if scheduler_type == "onecycle":
            total_steps = int(self.trainer.estimated_stepping_batches)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                total_steps=total_steps,
                pct_start=warmup_ratio,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
        elif scheduler_type == "cosine_warmup":
            max_epochs = int(self.trainer.max_epochs) if self.trainer.max_epochs else 10
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_epochs, eta_min=1e-6
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/epoch_loss",
                    "interval": "epoch",
                },
            }
        elif scheduler_type == "none":
            return {"optimizer": optimizer}
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    def configure_optimizers(self) -> Dict[str, Any] | None:
        """Configure optimizers and schedulers.
        
        Returns:
            Optimizer and scheduler configuration.
        """
        optimizer = self._build_optimizer()
        return self._build_scheduler(optimizer)
    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, Any], batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Test step.
        
        Args:
            batch: Tuple of (inputs, targets, _).
            batch_idx: Batch index.
            
        Returns:
            Tuple of (inputs, predictions, targets).
        """
        inputs, targets, _ = batch
        y_hat = self(inputs)
        targets = targets.float().view_as(y_hat)
        loss = self.criterion(y_hat, targets)
        self.log("test/loss", loss, prog_bar=True)
        return inputs, y_hat, targets
    
    def predict_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, Any], batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prediction step.
        
        Args:
            batch: Tuple of (inputs, targets, _).
            batch_idx: Batch index.
            
        Returns:
            Tuple of (inputs, predictions, targets).
        """
        inputs, targets, _ = batch
        y_hat = self(inputs)
        return inputs, y_hat, targets