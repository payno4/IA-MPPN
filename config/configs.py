
from dataclasses import dataclass
from config.IOConfig import IOConfig
from config.dataloader_config import DataloaderConfig
from config.dataset_config import DatasetConfig
from config.eval_optim_config import EvalOptimConfig
from config.experiment_config import ExperimentConfig
from config.model_configs.IA_ViT_configs import IA_VisionTransformerConfig
from config.model_configs.IA_MPPN_configs import IA_MPPNConfig
from config.model_configs.resnet_config import ResNetConfig
from config.trainer_config import TrainerConfig
from mlflow.tracking import MlflowClient


@dataclass
class Configs:
    io_config: IOConfig
    dataset_config: DatasetConfig
    dataloader_config: DataloaderConfig
    trainer_config: TrainerConfig
    eval_optim_config: EvalOptimConfig
    experiment_config: ExperimentConfig
    model_config: IA_VisionTransformerConfig | ResNetConfig | IA_MPPNConfig


def save_hyperparameters(config: Configs, experiment: MlflowClient):
    pass  # TODO implement saving hyperparameters to mlflow