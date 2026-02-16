"""Optuna optimization objective for MPPN model."""

import copy
from pathlib import Path

import lightning as pl
import optuna
from lightning.pytorch.loggers import MLFlowLogger
from optuna.integration import PyTorchLightningPruningCallback

from config.configs import Configs
from config.model_configs.MPPN_configs import MPPNConfig
from models.MPPN.lightning_module_mppn import MPPNLightningModule



def objective_MPPN(
    self: object, trial: optuna.Trial, config: Configs, datamodule: object | None = None
) -> float:
    """Optuna objective function for MPPN hyperparameter optimization.
    
    Args:
        self: Unused (kept for interface compatibility).
        trial: Optuna trial instance.
        config: Configuration object.
        datamodule: PyTorch Lightning DataModule.
        
    Returns:
        R2 score metric for the trial.
    """
    # Deep copy config to avoid modifying original
    config = copy.deepcopy(config)
    
    # Setup MLFlow logger
    logger = _setup_mlflow_logger(config, trial)
    
    # Suggest and apply hyperparameters
    _suggest_and_apply_hyperparameters(trial, config)
    
    # Log trial parameters
    _log_trial_parameters(trial)
    
    # Create model and trainer
    model = MPPNLightningModule(config)
    trainer = _build_trainer(config, trial, logger)
    
    # Train model
    trainer.fit(model, datamodule=datamodule)  # type: ignore
    
    # Return metric
    metric_name = "R2"
    return trainer.callback_metrics[f"val/metrics/{metric_name}"].item()


def _setup_mlflow_logger(config: Configs, trial: optuna.Trial) -> MLFlowLogger:
    """Setup MLFlow logger for the trial.
    
    Args:
        config: Configuration object.
        trial: Optuna trial instance.
        
    Returns:
        Configured MLFlowLogger instance.
    """
    mlflow_dir: Path = config.io_config.mlflow_dir
    return MLFlowLogger(
        experiment_name=config.eval_optim_config.optim.study_name,
        run_name=f"optuna_trial_{trial.number}",
        tracking_uri=f"sqlite:///{mlflow_dir.as_posix()}/mlflow.db",
    )


def _suggest_and_apply_hyperparameters(trial: optuna.Trial, config: Configs) -> None:
    """Suggest and apply hyperparameters to config.
    
    Args:
        trial: Optuna trial instance.
        config: Configuration object to modify.
    """
    feature_size = trial.suggest_categorical("feature_size", [64, 128, 256])
    representation_dim = trial.suggest_categorical("representation_dim", [128, 256, 512])
    
    model_config: MPPNConfig = config.model_config  # type: ignore
    model_config.feature_size = feature_size
    model_config.representation_dim = representation_dim


def _log_trial_parameters(trial: optuna.Trial) -> None:
    """Log trial parameters to console.
    
    Args:
        trial: Optuna trial instance.
    """
    print(f"Trial {trial.number} parameters:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")


def _build_trainer(config: Configs, trial: optuna.Trial, logger: MLFlowLogger) -> pl.Trainer:
    """Build PyTorch Lightning trainer with pruning callback.
    
    Args:
        config: Configuration object.
        trial: Optuna trial instance.
        logger: MLFlowLogger instance.
        
    Returns:
        Configured Trainer instance.
    """
    metric_name = "R2"
    return pl.Trainer(
        deterministic=True,
        accelerator=config.experiment_config.device,
        devices="auto",
        max_epochs=config.eval_optim_config.optim.max_train_epochs,
        enable_progress_bar=True,
        enable_checkpointing=False,
        logger=logger,
        callbacks=[
            PyTorchLightningPruningCallback(trial, monitor=f"val/metrics/{metric_name}")
        ],
    )