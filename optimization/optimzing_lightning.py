"""Lightning trainer optimization and hyperparameter tuning utilities."""

from typing import Any, Callable, Dict, List

import lightning as pl
import optuna
from lightning.pytorch.tuner import Tuner
from optuna.pruners import BasePruner
from optuna.samplers import BaseSampler

from config.configs import Configs
from models.IA_MPPN.lightning_module_ia_mppn import IA_MPPNLightningModule
from models.MPPN.lightning_module_mppn import MPPNLightningModule
from optimization.objective_IA_MPPN import objective_IA_MPPN
from optimization.objective_MPPN import objective_MPPN


class LightingOptimizer:
    """Lightning trainer optimizer for hyperparameter tuning and learning rate finding."""

    def __init__(
        self, config: Configs, datamodule: pl.LightningDataModule, **kwargs: Any
    ) -> None:
        """Initialize the optimizer.
        
        Args:
            config: Configuration object.
            datamodule: PyTorch Lightning DataModule.
            **kwargs: Additional keyword arguments (unused).
            
        Raises:
            ValueError: If model type is not supported.
        """
        self.config = config
        self.datamodule = datamodule
        self.search_space = self._build_search_space()
        self.trial: Callable[[optuna.Trial], float] | None = None

    def _build_search_space(self) -> Dict[str, List[Any]]:
        """Build search space based on model type.
        
        Returns:
            Dictionary of hyperparameter search space.
            
        Raises:
            ValueError: If model type is not supported.
        """
        model_type = self.config.eval_optim_config.optim.model_type
        if model_type == "IA-MPPN":
            return self._ia_mppn_search_space()
        elif model_type == "MPPN":
            return self._mppn_search_space()
        else:
            raise ValueError(f"Model {model_type} not supported for optimization.")

    def _ia_mppn_search_space(self) -> Dict[str, List[int]]:
        """Get IA-MPPN hyperparameter search space.
        
        Returns:
            Dictionary of IA-MPPN hyperparameter ranges.
        """
        return {
            "hidden_size": [256, 512],
            "mlp_dim": [2, 4],
            "num_layers": [6, 8, 10],
        }

    def _mppn_search_space(self) -> Dict[str, List[int]]:
        """Get MPPN hyperparameter search space.
        
        Returns:
            Dictionary of MPPN hyperparameter ranges.
        """
        return {
            "feature_size": [64, 128, 256],
            "representation_dim": [128, 256, 512],
        }

    def optimize(self) -> None:
        """Run hyperparameter optimization using Optuna."""
        # Setup pruner and sampler
        pruner = self._build_pruner()
        sampler = self._build_sampler()
        
        # Setup study database
        db_path = self._setup_database_path()
        
        # Create study
        study = self._create_study(pruner, sampler, db_path)
        
        # Setup objective function
        self._setup_objective_function()
        
        # Run optimization
        if self.trial is not None:
            study.optimize(self.trial, gc_after_trial=True)
        
        # Print results
        self._print_results(study)

    def _build_pruner(self) -> BasePruner:
        """Build pruner for trial pruning.
        
        Returns:
            Configured pruner instance.
        """
        return optuna.pruners.MedianPruner(
            n_startup_trials=10, n_warmup_steps=200, interval_steps=10
        )

    def _build_sampler(self) -> BaseSampler:
        """Build sampler for hyperparameter sampling.
        
        Returns:
            Configured sampler instance.
        """
        return optuna.samplers.GridSampler(self.search_space)

    def _setup_database_path(self) -> str:
        """Setup and create database path for study.
        
        Returns:
            Path to study database.
        """
        db_path = (
            self.config.io_config.optimizer_dir
            / self.config.eval_optim_config.optim.study_name
        )
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return str(db_path)

    def _create_study(self, pruner: BasePruner, sampler: BaseSampler, db_path: str) -> optuna.Study:
        """Create Optuna study.
        
        Args:
            pruner: Configured pruner.
            sampler: Configured sampler.
            db_path: Path to study database.
            
        Returns:
            Optuna study instance.
        """
        return optuna.create_study(
            direction="maximize",
            storage=f"sqlite:///{db_path}.db",
            sampler=sampler,
            study_name=self.config.eval_optim_config.optim.study_name,
            load_if_exists=True,
            pruner=pruner,
        )

    def _setup_objective_function(self) -> None:
        """Setup objective function based on model type.
        
        Raises:
            ValueError: If model type is not supported.
        """
        model_type = self.config.eval_optim_config.optim.model_type
        if model_type == "IA-MPPN":
            self.trial = lambda trial: objective_IA_MPPN(
                self, trial, self.config, self.datamodule
            )
        elif model_type == "MPPN":
            self.trial = lambda trial: objective_MPPN(
                self, trial, self.config, self.datamodule
            )
        else:
            raise ValueError(f"Model {model_type} not supported for optimization.")

    def _print_results(self, study: optuna.Study) -> None:
        """Print optimization results.
        
        Args:
            study: Optuna study instance.
        """
        completed_trials = study.get_trials(
            deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)
        )
        pruned_trials = study.get_trials(
            deepcopy=False, states=(optuna.trial.TrialState.PRUNED,)
        )
        
        print("Best hyperparameters:", study.best_params)
        print("Best value:", study.best_value)
        print("Number of trials:", len(study.trials))
        print("Number of finished trials:", len(completed_trials))
        print("Number of pruned trials:", len(pruned_trials))

    def get_best_lr(self) -> float | None:
        """Find optimal learning rate using LR finder.
        
        Returns:
            Suggested learning rate or None if not found.
        """
        model = self._build_model_for_lr_finding()
        trainer = self._build_trainer_for_lr_finding()
        
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(
            model,
            datamodule=self.datamodule,
            min_lr=1e-8,
            max_lr=0.001,
        )
        
        if lr_finder is not None:
            suggested_lr = lr_finder.suggestion()
            print(f"Suggested learning rate: {suggested_lr}")
            return suggested_lr
        return None

    def _build_model_for_lr_finding(self) -> IA_MPPNLightningModule | MPPNLightningModule:
        """Build model for learning rate finding.
        
        Returns:
            Lightning module instance.
        """
        model_type = self.config.eval_optim_config.optim.model_type
        if model_type == "IA-MPPN":
            return IA_MPPNLightningModule(self.config)
        else:
            return MPPNLightningModule(self.config)

    def _build_trainer_for_lr_finding(self) -> pl.Trainer:
        """Build trainer for learning rate finding.
        
        Returns:
            Configured Trainer instance.
        """
        return pl.Trainer(
            deterministic=True,
            accelerator=self.config.experiment_config.device,
            devices="auto",
            max_epochs=self.config.eval_optim_config.optim.max_train_epochs,
            enable_progress_bar=True,
            enable_checkpointing=False,
        )