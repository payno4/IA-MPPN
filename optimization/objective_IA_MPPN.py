"""Optuna optimization objective for IA-MPPN model."""

import copy
from pathlib import Path

import lightning as pl
import optuna
from lightning.pytorch.loggers import MLFlowLogger
from optuna.integration import PyTorchLightningPruningCallback

from config.configs import Configs
from config.model_configs.IA_MPPN_configs import IA_MPPNConfig
from models.IA_MPPN.lightning_module_ia_mppn import IA_MPPNLightningModule



def objective_IA_MPPN(
    self: object, trial: optuna.Trial, config: Configs, datamodule: object | None = None
) -> float:
    """Optuna objective function for IA-MPPN hyperparameter optimization.
    
    Args:
        self: Unused (kept for interface compatibility).
        trial: Optuna trial instance.
        config: Configuration object.
        datamodule: PyTorch Lightning DataModule.
        
    Returns:
        R2 score metric for the trial.
        
    Raises:
        optuna.TrialPruned: If trial parameters are invalid.
    """
    # Deep copy config to avoid modifying original
    config = copy.deepcopy(config)
    
    # Setup MLFlow logger
    logger = _setup_mlflow_logger(config, trial)
    
    # Suggest hyperparameters
    trial_params = _suggest_hyperparameters(trial, config)
    
    # Validate and apply hyperparameters
    _validate_and_apply_hyperparameters(config, trial_params, trial)
    
    # Log trial parameters
    _log_trial_parameters(trial)
    
    # Create model and trainer
    model = IA_MPPNLightningModule(config)
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


def _suggest_hyperparameters(trial: optuna.Trial, config: Configs) -> dict:
    """Suggest hyperparameters for the trial.
    
    Args:
        trial: Optuna trial instance.
        config: Configuration object.
        
    Returns:
        Dictionary of suggested hyperparameters.
    """
    num_heads = config.model_config.perspective_model_config.num_heads  # type: ignore
    
    return {
        "hidden_size": trial.suggest_categorical("hidden_size", [256, 512]),
        "mlp_dim": trial.suggest_categorical("mlp_dim", [2, 4]),
        "num_heads": num_heads,
        "num_layers": trial.suggest_categorical("num_layers", [6, 8, 10]),
    }


def _validate_and_apply_hyperparameters(
    config: Configs, params: dict, trial: optuna.Trial
) -> None:
    """Validate and apply suggested hyperparameters to config.
    
    Args:
        config: Configuration object to modify.
        params: Suggested hyperparameters.
        trial: Optuna trial instance (for pruning).
        
    Raises:
        optuna.TrialPruned: If parameters are invalid.
    """
    hidden_size = params["hidden_size"]
    num_heads = params["num_heads"]
    mlp_dim = params["mlp_dim"]
    num_layers = params["num_layers"]
    
    # Validate divisibility
    if hidden_size % num_heads != 0:
        raise optuna.TrialPruned("hidden_size must be divisible by num_heads")
    
    # Apply to perspective model config
    model_config: IA_MPPNConfig = config.model_config  # type: ignore
    model_config.perspective_model_config.hidden_size = hidden_size
    model_config.perspective_model_config.mlp_dim = mlp_dim * hidden_size
    model_config.perspective_model_config.num_heads = num_heads
    model_config.perspective_model_config.num_layers = num_layers
    
    # Apply to relation model config
    model_config.relation_model_config.hidden_size = hidden_size
    model_config.relation_model_config.mlp_dim = mlp_dim * hidden_size
    model_config.relation_model_config.num_attention_heads = num_heads
    model_config.relation_model_config.num_hidden_layers = num_layers


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