"""Machine learning pipeline for event log analysis with model training and evaluation."""

import json
from typing import Any, Dict, List

import numpy as np
import torch
from lightning.pytorch import seed_everything

from config.IOConfig import IOConfig
from config.config_factory import ConfigFactory
from config.configs import Configs
from config.dataloader_config import DataloaderConfig
from config.dataset_config import DatasetConfig
from config.eval_optim_config import EvalOptimConfig
from config.experiment_config import ExperimentConfig
from config.model_configs.IA_MPPN_configs import IA_MPPNConfig
from config.model_configs.MPPN_configs import MPPNConfig
from config.model_configs.resnet_config import ResNetConfig
from config.trainer_config import TrainerConfig
from dataset_preprocess.new_assemble import DataAssembler  
from evaluation.performance_eval import PerformanceEvaluator
from evaluation.visualisation import Visualisation
from dataset_loader.lightning_datamodule import DataModuleHeatmap 
from models.trainers.IA_MPPN_training import IA_MPPNTrainer
from models.trainers.MPPN_training import MPPNTrainer
from optimization.optimzing_lightning import LightingOptimizer
from pipeline.base_pipeline import BasePipeline
from dataset_preprocess.xes_process import read_xes_file
from evaluation.metrics import Metrics

class Pipeline(BasePipeline):
    """Main pipeline for event log processing, model training, and evaluation."""

    def __init__(self, config_paths: Dict[str, str]) -> None:
        """Initialize the pipeline with configuration paths.
        
        Args:
            config_paths: Dictionary mapping config names to file paths.
            
        Raises:
            ValueError: If model type is not recognized.
        """
        # Load all configurations
        experiment_config = self._load_experiment_config(config_paths)
        io_config = self._load_io_config(config_paths)
        dataset_config = self._load_dataset_config(config_paths)
        dataloader_config = self._load_dataloader_config(config_paths)
        trainer_config = self._load_trainer_config(config_paths)
        eval_optim_config = self._load_eval_optim_config(config_paths)
        model_config = self._load_model_config(config_paths, experiment_config.model_name)
        
        # Create consolidated config
        self.configs: Configs = Configs(
            io_config=io_config,
            dataset_config=dataset_config,
            dataloader_config=dataloader_config,
            trainer_config=trainer_config,
            eval_optim_config=eval_optim_config,
            experiment_config=experiment_config,
            model_config=model_config,  # type: ignore
        )
        
        # Initialize parent and device
        self.console_messages = experiment_config.console_messages
        self.device = self._resolve_device(experiment_config)
        BasePipeline.__init__(self, self.console_messages)
        
        # Setup seed
        self._setup_seed(experiment_config.seed)
        self._console_message(f"Your device: {self.device}")
        
        # Setup directories
        self.directory_data = io_config.raw_data_dir
        self.directory_h5 = io_config.h5_dir
        self.directory_analysis = io_config.analysis_dir
        self.h5_dir = io_config.h5_dir
        
        # Initialize helper classes
        self.config_factory = ConfigFactory(
            dataloader_config, str(self.directory_h5), verbose=self.console_messages  # type: ignore
        )
        
        # Initialize dataset
        self.dataset: Any = None
        self.synthetic_data = False
        self.matrix_states: Any = None
        self.event_log_df: Any = None
        self.trace_keys: Any = None

    @staticmethod
    def _load_json_config(file_path: str | None) -> Dict[str, Any]:
        """Load JSON configuration file.
        
        Args:
            file_path: Path to configuration file.
            
        Returns:
            Dictionary containing configuration or empty dict if not found.
        """
        if file_path is None:
            return {}
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def _load_experiment_config(self, config_paths: Dict[str, str]) -> ExperimentConfig:
        """Load experiment configuration.
        
        Args:
            config_paths: Dictionary mapping config names to file paths.
            
        Returns:
            Loaded ExperimentConfig instance.
        """
        return ExperimentConfig.from_dict(  # type: ignore
            self._load_json_config(config_paths.get("experiment_config"))
        )
    
    def _load_io_config(self, config_paths: Dict[str, str]) -> IOConfig:
        """Load IO configuration.
        
        Args:
            config_paths: Dictionary mapping config names to file paths.
            
        Returns:
            Loaded IOConfig instance.
        """
        return IOConfig.from_dict(  # type: ignore
            self._load_json_config(config_paths.get("io_config"))
        )
    
    def _load_dataset_config(self, config_paths: Dict[str, str]) -> DatasetConfig:
        """Load dataset configuration.
        
        Args:
            config_paths: Dictionary mapping config names to file paths.
            
        Returns:
            Loaded DatasetConfig instance.
        """
        return DatasetConfig.from_dict(  # type: ignore
            self._load_json_config(config_paths.get("dataset_config"))
        )
    
    def _load_dataloader_config(self, config_paths: Dict[str, str]) -> DataloaderConfig:
        """Load dataloader configuration.
        
        Args:
            config_paths: Dictionary mapping config names to file paths.
            
        Returns:
            Loaded DataloaderConfig instance.
        """
        return DataloaderConfig.from_dict(  # type: ignore
            self._load_json_config(config_paths.get("dataloader_config"))
        )
    
    def _load_trainer_config(self, config_paths: Dict[str, str]) -> TrainerConfig:
        """Load trainer configuration.
        
        Args:
            config_paths: Dictionary mapping config names to file paths.
            
        Returns:
            Loaded TrainerConfig instance.
        """
        return TrainerConfig.from_dict(  # type: ignore
            self._load_json_config(config_paths.get("trainer_config"))
        )
    
    def _load_eval_optim_config(self, config_paths: Dict[str, str]) -> EvalOptimConfig:
        """Load evaluation and optimization configuration.
        
        Args:
            config_paths: Dictionary mapping config names to file paths.
            
        Returns:
            Loaded EvalOptimConfig instance.
        """
        return EvalOptimConfig.from_dict(  # type: ignore
            self._load_json_config(config_paths.get("eval_optim_config"))
        )
    
    def _load_model_config(
        self, config_paths: Dict[str, str], model_name: str
    ) -> ResNetConfig | IA_MPPNConfig | MPPNConfig:
        """Load model-specific configuration.
        
        Args:
            config_paths: Dictionary mapping config names to file paths.
            model_name: Name of the model.
            
        Returns:
            Loaded model configuration.
            
        Raises:
            ValueError: If model type is not recognized.
        """
        if model_name == "resnet":
            return ResNetConfig.from_dict(  # type: ignore
                self._load_json_config(config_paths.get("resnet_config"))
            )
        elif model_name == "IA-MPPN":
            return IA_MPPNConfig.from_dict(  # type: ignore
                self._load_json_config(config_paths.get("IA_MPPN_config"))
            )
        elif model_name == "MPPN":
            return MPPNConfig.from_dict(  # type: ignore
                self._load_json_config(config_paths.get("MPPN_config"))
            )
        else:
            raise ValueError(f"Model '{model_name}' not recognized.")
    
    def _resolve_device(self, experiment_config: ExperimentConfig) -> str:
        """Resolve device for training.
        
        Args:
            experiment_config: Experiment configuration.
            
        Returns:
            Device string ('cuda' or 'cpu').
        """
        device = experiment_config.device
        if device == "lookup":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            experiment_config.device = device
        return device
    
    def _setup_seed(self, seed: int | str) -> None:
        """Setup random seed for reproducibility.
        
        Args:
            seed: Seed value or "None" string.
        """
        if seed != "None" and seed is not None:
            seed_everything(seed)  # type: ignore
            self._console_message(f"Seed set to: {seed}")
        else:
            self._console_message("No seed set for this run.")

    def read(self, xes_file_name: str) -> None:
        """Read XES file and extract event log data.
        
        Args:
            xes_file_name: Name of the XES file to read.
        """
        self.matrix_states, self.event_log_df, self.trace_keys = read_xes_file(
            self.directory_data, xes_file_name
        )

    def create_data(
        self,
        attrs_list: List[str],
        insertion_list: List[str] | None = None,
        groupby: str = "case:concept:name",
    ) -> None:
        """Create and assemble dataset from event logs.
        
        Args:
            attrs_list: List of attributes to include.
            insertion_list: List of attributes to insert. If None, uses attrs_list.
            groupby: Column name to group by. Defaults to "case:concept:name".
            
        Raises:
            AssertionError: If groupby attribute is in attrs_list or insertion_list.
        """
        # Validate inputs
        assert (
            groupby not in attrs_list
        ), f"The groupby attribute '{groupby}' should not be in the attributes list."
        if insertion_list is not None:
            assert (
                groupby not in insertion_list
            ), f"The groupby attribute '{groupby}' should not be in the insertion list."
        
        # Setup data assembler
        da = DataAssembler(self.configs)
        
        # Prepare groups and split data
        groups = self.event_log_df[groupby].unique()
        max_log_length = self.event_log_df.groupby(groupby).size().max()
        rng = np.random.default_rng(self.configs.experiment_config.seed)
        rng.shuffle(groups)
        
        # Get split indices (handle 2 or 3 element tuple)
        split_idxs = self.configs.dataset_config.split_idxs
        train_idx, val_idx = split_idxs[0], split_idxs[1]
        
        train_groups, val_groups, test_groups = self._split_groups(
            groups, (train_idx, val_idx)
        )
        
        # Create dataset splits
        train_df = self.event_log_df[self.event_log_df[groupby].isin(train_groups)].reset_index(
            drop=True
        )
        val_df = self.event_log_df[self.event_log_df[groupby].isin(val_groups)].reset_index(
            drop=True
        )
        test_df = self.event_log_df[self.event_log_df[groupby].isin(test_groups)].reset_index(
            drop=True
        )
        
        # Assemble datasets
        for df, mode in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
            da.assemble_data(
                event_logs_df=df,
                attributes_list=attrs_list,
                groupby=groupby,
                mode=mode,
                max_len_log=max_log_length,
            )
        
        # Create input datasets
        if insertion_list is None:
            insertion_list = attrs_list
        self.create_new_dataset(insertion_list, group_name="input_1", da=da)
    
    def _split_groups(
        self, groups: np.ndarray, split_idxs: tuple[float, float]
    ) -> tuple[set[Any], set[Any], set[Any]]:
        """Split groups into train, validation, and test sets.
        
        Args:
            groups: Array of group identifiers.
            split_idxs: Tuple of (train_ratio, val_ratio).
            
        Returns:
            Tuple of (train_groups, val_groups, test_groups).
        """
        train_ratio, val_ratio = split_idxs
        train_idx = int(train_ratio * len(groups))
        val_idx = int(val_ratio * len(groups) + train_idx)
        
        return (
            set(groups[:train_idx]),
            set(groups[train_idx:val_idx]),
            set(groups[val_idx:]),
        )
    
    def create_new_dataset(
        self,
        insertion_list: List[str],
        group_name: str,
        da: DataAssembler | None = None,
    ) -> None:
        """Create input datasets for all data splits.
        
        Args:
            insertion_list: List of attributes to insert.
            group_name: Name of the group in the dataset.
            da: DataAssembler instance. If None, creates a new one.
        """
        if da is None:
            da = DataAssembler(self.configs)  # type: ignore
        
        for mode in ["train", "val", "test"]:
            da.create_input_dataset(  # type: ignore
                mode=mode, insertion_attrs=insertion_list, group_name=group_name
            )   
    
    def load_dataset(self) -> None:
        """Load and build datasets from configuration."""
        train_dataset, val_dataset, test_dataset = self.config_factory.build_dataset()
        datasets = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }
        self.dataset = DataModuleHeatmap(
            datasets=datasets, config=self.configs.dataloader_config # type: ignore
        )
            

    def analyze_hyperparameters(self) -> None:
        """Run hyperparameter optimization using Optuna.
        
        Raises:
            AssertionError: If dataset is not loaded.
        """
        assert (
            self.dataset is not None
        ), "No dataset provided, please load a dataset first."
        
        optimizer = LightingOptimizer(self.configs, self.dataset)
        optimizer.optimize()
                        
    def train_model(self) -> None:
        """Train the model using the loaded dataset.
        
        Raises:
            ValueError: If dataset is not loaded or model type is not supported.
        """
        if self.dataset is None:
            raise ValueError("No dataset provided, please load a dataset first.")
        
        model_name = self.configs.experiment_config.model_name
        trainer = self._build_trainer(model_name)
        trainer.train(ckpt_path="")  # type: ignore
    
    def _build_trainer(
        self, model_name: str
    ) ->  IA_MPPNTrainer | MPPNTrainer:
        """Build trainer for the specified model.
        
        Args:
            model_name: Name of the model.
            
        Returns:
            Configured trainer instance.
            
        Raises:
            ValueError: If model type is not supported.
        """
        if model_name == "IA-MPPN":
            return IA_MPPNTrainer(config=self.configs, dataset=self.dataset)
        elif model_name == "MPPN":
            return MPPNTrainer(config=self.configs, dataset=self.dataset)
        else:
            raise ValueError(f"Model '{model_name}' not supported for training.")

    def evaluate_model(
        self, interpre: bool = True, vis: bool = False, per: bool = False
    ) -> None:
        """Evaluate model using various metrics and visualizations.
        
        Args:
            interpre: Whether to calculate interpretability metrics. Defaults to True.
            vis: Whether to generate visualizations. Defaults to False.
            per: Whether to evaluate performance. Defaults to False.
        """
        if interpre:
            quantus_metrics = Metrics(configs=self.configs)
            quantus_metrics.calculate_metrics(self.dataset)
        
        model_path = self.configs.eval_optim_config.evaluation.model_path_1
        
        if vis:
            visualization = Visualisation(model_path=model_path, configs=self.configs)
            visualization.visualize(dataset=self.dataset)
        
        if per:
            perf_eval = PerformanceEvaluator(
                model_path=model_path,
                model_type=self.configs.experiment_config.model_name,
                configs=self.configs,
            )
            perf_eval.evaluate_with_bootstrap(self.dataset)
