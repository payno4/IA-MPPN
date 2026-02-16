"""Lightning module for IA-MPPN model training and evaluation."""

from typing import Any, Dict, List, Tuple

import lightning as pl
import torch
import torchmetrics
from lightning.pytorch.loggers import MLFlowLogger
from mlflow.tracking.client import MlflowClient
from torch import nn
from torch.nn import functional as F

from config.configs import Configs
from config.model_configs.IA_MPPN_configs import IA_MPPNConfig
from models.IA_MPPN.ia_mppn import CollectedAttentions, IA_MPPN
from models.IA_ViT.losses import MMDLossRegression
from pipeline.utils import get_config_details

class IA_MPPNLightningModule(pl.LightningModule):
    """Lightning module for IA-MPPN model training with attention-based interpretability."""

    def __init__(
        self, config: Configs, model: IA_MPPN | None = None, learning_rate: float | None = None
    ) -> None:
        """Initialize the lightning module.
        
        Args:
            config: Configuration object.
            model: Optional pre-built IA_MPPN model. If None, creates new model.
            learning_rate: Optional learning rate override.
        """
        assert isinstance(
            config.model_config, IA_MPPNConfig
        ), "config.model_config must be an instance of IA_MPPNConfig for IA-MPPN model."
        super().__init__()
        self.config = config
        self.ablation_mode = config.model_config.ablation_mode
        self.learning_rate = learning_rate or config.trainer_config.lr

        # Initialize model
        input_shape = tuple(config.dataloader_config.input_shape)  # type: ignore
        self.model = model or IA_MPPN(config.model_config, input_shape=input_shape)  # type: ignore

        # Storage for outputs
        self.val_outs: List[Dict[str, torch.Tensor]] = []
        self.test_outs: List[Dict[str, torch.Tensor]] = []
        self.train_outs: List[Dict[str, torch.Tensor]] = []

        # Loss function
        self.criterion = self._build_criterion()
        
        # Metrics
        self.metrics = ["MAE", "MSE", "R2"]
        self._build_metrics()

        # MMD loss for distillation
        self.mmd_loss_fn = MMDLossRegression(sigma=None)

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
        
        # Only add perspective-specific tags if using IA_MPPNConfig
        if not isinstance(self.config.model_config, IA_MPPNConfig):
            return
            
        tags = {
            "Model": self.config.experiment_config.model_name,
            "Ablationmode": str(self.ablation_mode),
            "Dataset": str(self.config.dataloader_config.h5_file_name),
            "hidden size": str(self.config.model_config.perspective_model_config.hidden_size),  # type: ignore
            "num layers": str(self.config.model_config.perspective_model_config.num_layers),  # type: ignore
            "num heads": str(self.config.model_config.perspective_model_config.num_heads),  # type: ignore
            "patch size": str(self.config.model_config.perspective_model_config.patches["size"]),  # type: ignore
            "mlp dim": str(self.config.model_config.perspective_model_config.mlp_dim),  # type: ignore
        }
        for key, value in tags.items():
            experiment.set_tag(self.logger.run_id, key, value)


    def on_train_start(self) -> None:
        """Log learning rate at training start."""
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        print(f"Used lr: {lr}")

    def forward(
        self, x: torch.Tensor, evaluation: bool
    ) -> Tuple[torch.Tensor, ...] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the model.
        
        Args:
            x: Input tensor.
            evaluation: Whether in evaluation mode (returns attention).
            
        Returns:
            During training: (logits, inter_logits, inter_features).
            During evaluation: (logits, inter_logits, inter_features, attention).
        """
        if evaluation:
            logits, inter_logits, inter_features, ca = self.model(x, evaluation=True)
            return logits, inter_logits, inter_features, ca
        else:
            logits, inter_logits, inter_features = self.model(x, evaluation=False)
            return logits, inter_logits, inter_features

    def _compute_relation_mmd_loss(self, ca: CollectedAttentions | None) -> torch.Tensor:
        """Compute MMD loss for relation attention.
        
        Args:
            ca: Collected attention weights.
            
        Returns:
            Weighted MMD loss for relation attention.
        """
        if ca is None or ca.relation_attention is None:
            return torch.tensor(0.0)
            
        atts = ca.relation_attention.attn_weights  # [layers, B, heads, N, N]
        new_atts = ca.relation_attention.new_atts  # [B, N, N] w/o cls token
        student_patches = new_atts.mean(dim=1)  # [B, N]  # type: ignore
        
        old_atts = atts[-1].mean(dim=1)  # [B, N, N]  # type: ignore
        teacher_patches = old_atts[:, 1:, 1:].mean(dim=1)  # w/o cls token [B, T]
        
        rel_patch_mmd_loss = self.mmd_loss_fn(teacher_patches.detach(), student_patches)
        self.log("train/relation_patch_mmd_loss", rel_patch_mmd_loss, prog_bar=False)
        return 0.05 * rel_patch_mmd_loss

    def _compute_perspective_mmd_loss(self, ca: CollectedAttentions | None) -> torch.Tensor:
        """Compute MMD loss for perspective attentions.
        
        Args:
            ca: Collected attention weights.
            
        Returns:
            Average MMD loss across all perspectives.
        """
        if ca is None or ca.perspective_attentions is None:
            return torch.tensor(0.0)
            
        mmd_vals = []
        for p_idx, p_attn in enumerate(ca.perspective_attentions):
            atts = p_attn.attn_weights  # [layers, B, heads, N, N]
            new_atts = p_attn.new_atts  # [B, N, N]
            new_atts = new_atts.mean(dim=1)  # [B, N]  # type: ignore
            
            old_atts = atts[-1].mean(dim=1)  # [B, N, N]  # type: ignore
            teacher_patches = old_atts[:, 1:, 1:].mean(dim=1)  # w/o cls token [B, T]
            student_patches = new_atts  # [B, N]
            
            patch_mmd_loss = self.mmd_loss_fn(teacher_patches.detach(), student_patches)
            self.log(f"train/perspective_{p_idx}_patch_mmd_loss", patch_mmd_loss, prog_bar=False)
            mmd_vals.append(patch_mmd_loss)
        
        p_mmd_loss = torch.stack(mmd_vals).mean()
        self.log("train/perspective_mmd_loss", p_mmd_loss, prog_bar=False)
        return p_mmd_loss
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, Any], batch_idx: int) -> torch.Tensor:
        """Training step with task, distillation, and MMD losses.
        
        Args:
            batch: Tuple of (inputs, targets, _).
            batch_idx: Batch index.
            
        Returns:
            Total loss.
        """
        inputs, targets, _ = batch
        logits, inter_logits, _, ca = self(inputs, evaluation=True)
        targets = targets.float().view_as(logits)
        
        # Compute losses
        task_loss = self.criterion(logits, targets)
        kd_loss = nn.SmoothL1Loss()(inter_logits, logits.detach())  # Knowledge distillation
        rel_mmd_loss = self._compute_relation_mmd_loss(ca)
        persp_mmd_loss = self._compute_perspective_mmd_loss(ca)
        mmd_loss = rel_mmd_loss + persp_mmd_loss
        
        # Total loss
        loss = task_loss + kd_loss + mmd_loss
        
        # Logging
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/mmd_loss", mmd_loss, prog_bar=False)
        self.log("train/task_loss", task_loss, prog_bar=True)
        self.log("train/kd_loss", kd_loss, prog_bar=False)
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
        logits, _, _ = self(inputs, evaluation=False)
        targets = targets.float().view_as(logits)
        loss = self.criterion(logits, targets)
        
        self._compute_metrics(logits, targets, "val")
        self.val_outs.append({
            "val_loss": loss.detach().float().cpu(),
            "predictions": logits.detach().float().cpu(),
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, CollectedAttentions]:
        """Test step.
        
        Args:
            batch: Tuple of (inputs, targets, _).
            batch_idx: Batch index.
            
        Returns:
            Tuple of (inputs, predictions, targets, inter_logits, attention_weights).
        """
        inputs, targets, _ = batch
        logits, inter_logits, _, ca = self(inputs, evaluation=True)
        targets = targets.float().view_as(logits)
        loss = self.criterion(logits, targets)
        self.log("test_loss", loss, prog_bar=True)
        return inputs, logits, targets, inter_logits, ca

    def predict_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, Any], batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, CollectedAttentions]:
        """Prediction step.
        
        Args:
            batch: Tuple of (inputs, targets, _).
            batch_idx: Batch index.
            
        Returns:
            Tuple of (inputs, targets, predictions, inter_logits, attention_weights).
        """
        inputs, targets, _ = batch
        logits, inter_logits, _, ca = self(inputs, evaluation=True)
        return inputs, targets, logits, inter_logits, ca