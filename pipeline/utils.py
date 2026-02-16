"""Utility functions for pipeline configuration extraction and analysis."""

from typing import Any, Dict

from config.configs import Configs
from config.model_configs.IA_MPPN_configs import IA_MPPNConfig
from config.model_configs.IA_ViT_configs import IA_VisionTransformerConfig
from config.model_configs.MPPN_configs import MPPNConfig
from config.trainer_config import TrainerConfig
from config.dataloader_config import DataloaderConfig


def get_config_details(configs: Configs) -> Dict[str, Any]:
    """Extract and format configuration details from Configs object.
    
    Args:
        configs: Configs object containing all pipeline configurations.
        
    Returns:
        Dictionary with formatted configuration details.
        
    Raises:
        AssertionError: If configs is not a valid Configs object.
        ValueError: If model name is not recognized.
    """
    assert isinstance(configs, Configs), "Please provide a valid Configs object."

    model_config = configs.model_config
    trainer_config = configs.trainer_config
    experiment_config = configs.experiment_config
    dataloader_config = configs.dataloader_config
    
    configs_out: Dict[str, Any] = {}

    if experiment_config.model_name == "IA-MPPN":
        assert isinstance(model_config, IA_MPPNConfig), "Model config must be IA_MPPNConfig"
        model_configs_out = {
            "perspectiv component": {
                "hidden size": model_config.perspective_model_config.hidden_size,
                "num hidden layers": model_config.perspective_model_config.num_layers,
                "mlp dim": model_config.perspective_model_config.mlp_dim,
                "num attention heads": model_config.perspective_model_config.num_heads,
                "patch size": model_config.perspective_model_config.patches["size"],
                "dropout rate": model_config.perspective_model_config.attention_dropout_rate,
                "attention dropout rate": model_config.perspective_model_config.attention_dropout_rate
            },
            "relation component": {
                "hidden size": model_config.relation_model_config.hidden_size,
                "num hidden layers": model_config.relation_model_config.num_hidden_layers,
                "mlp dim": model_config.relation_model_config.mlp_dim,
                "num attention heads": model_config.relation_model_config.num_attention_heads,
                "dropout rate": model_config.relation_model_config.attention_dropout_rate,
                "attention dropout rate": model_config.relation_model_config.attention_dropout_rate
            }
        }
        configs_out["model configurations"] = model_configs_out
    elif experiment_config.model_name == "MP-IA-ViTransformer":
        assert isinstance(model_config, IA_VisionTransformerConfig), "Model config must be IA_VisionTransformerConfig"
        model_configs_out = {
            "hidden size": model_config.hidden_size,
            "num hidden layers": model_config.num_layers,
            "mlp dim": model_config.mlp_dim,
            "num attention heads": model_config.num_heads,
            "patch size": model_config.patches["size"],
            "dropout rate": model_config.attention_dropout_rate,
            "attention dropout rate": model_config.attention_dropout_rate
        }
        configs_out["model configurations"] = model_configs_out
    elif experiment_config.model_name == "MPPN":
        assert isinstance(model_config, MPPNConfig), "Model config must be MPPNConfig"
        model_configs_out = {
            "feature size": model_config.feature_size,
            "repr. dim": model_config.representation_dim,
        }
        configs_out["model configurations"] = model_configs_out
    else:
        raise ValueError(f"Model name {experiment_config.model_name} not recognized for configuration extraction.")
    
    trainer_config_out = {
        "training epochs": trainer_config.epochs,
        "learning rate": trainer_config.lr,
        "weight decay": trainer_config.weight_decay,
        "scheduler": trainer_config.scheduler,
        "warmup ratio": trainer_config.warmup_ratio,
        "optimizer": trainer_config.optimizer,
        "criterion": trainer_config.criterion,
        "beta1": trainer_config.beta1,
        "beta2": trainer_config.beta2
    }
    configs_out["trainer configurations"] = trainer_config_out
    
    dataloader_config_out = {
        "train dataloader": {
            "num workers": dataloader_config.train_dataloader.num_workers,
            "batch size": dataloader_config.train_dataloader.batch_size,
            "shuffle": dataloader_config.train_dataloader.shuffle,
            "pin memory": dataloader_config.train_dataloader.pin_memory,
            "drop last": dataloader_config.train_dataloader.drop_last,
            "persistent workers": dataloader_config.train_dataloader.persistent_workers
        },
        "test dataloader": {
            "num workers": dataloader_config.test_dataloader.num_workers,
            "batch size": dataloader_config.test_dataloader.batch_size,
            "shuffle": dataloader_config.test_dataloader.shuffle,
            "pin memory": dataloader_config.test_dataloader.pin_memory,
            "drop last": dataloader_config.test_dataloader.drop_last,
            "persistent workers": dataloader_config.test_dataloader.persistent_workers
        },
        "validation dataloader": {
            "num workers": dataloader_config.validation_dataloader.num_workers,
            "batch size": dataloader_config.validation_dataloader.batch_size,
            "shuffle": dataloader_config.validation_dataloader.shuffle,
            "pin memory": dataloader_config.validation_dataloader.pin_memory,
            "drop last": dataloader_config.validation_dataloader.drop_last,
            "persistent workers": dataloader_config.validation_dataloader.persistent_workers
        }
    }
    configs_out["dataloader configurations"] = dataloader_config_out
    return configs_out


