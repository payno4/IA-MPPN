from __future__ import annotations

from pathlib import Path
from typing import Tuple

import lightning as pl
import torch

from config.configs import Configs


def load_lightning_model(
    model_path: str,
    model_type: str,
    config: Configs,
    lightning_datamodule: pl.LightningDataModule | None,
) -> Tuple[torch.nn.Module, pl.LightningModule | None]:
    """Load model weights and optionally wrap in LightningModule."""
    input_shape = _get_input_shape(config)
    num_perspectives = config.dataloader_config.input_shape[2]

    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        raise FileNotFoundError(f"Model file not found at {str(model_path_obj)}")

    ckpt = torch.load(model_path_obj, map_location=config.experiment_config.device, weights_only=False)
    cleaned_state_dict = _clean_state_dict(ckpt["state_dict"])

    model, lightning_model = _build_model(model_type, config, input_shape, num_perspectives, lightning_datamodule)
    missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
    print(f"{model_type} missing keys: {missing_keys}")
    print(f"{model_type} unexpected keys: {unexpected_keys}")

    return model, lightning_model


def _get_input_shape(config: Configs) -> Tuple[int, int, int]:
    H, W, P = (
        config.dataloader_config.input_shape[0],
        config.dataloader_config.input_shape[1],
        config.dataloader_config.input_shape[2],
    )
    return H, W, P


def _clean_state_dict(state_dict: dict) -> dict:
    return {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}


def _build_model(
    model_type: str,
    config: Configs,
    input_shape: Tuple[int, int, int],
    num_perspectives: int,
    lightning_datamodule: pl.LightningDataModule | None,
) -> Tuple[torch.nn.Module, pl.LightningModule | None]:
    match model_type:
        case "IA-MPPN":
            from models.IA_MPPN.ia_mppn import IA_MPPN
            from models.IA_MPPN.lightning_module_ia_mppn import IA_MPPNLightningModule

            model = IA_MPPN(config.model_config, input_shape) # type: ignore
            lightning_model = (
                None
                if lightning_datamodule is not None
                else IA_MPPNLightningModule(config, model=model)
            )
            return model, lightning_model

        case "MPPN":
            from models.MPPN.lightning_module_mppn import MPPNLightningModule
            from models.MPPN.original_MPPN import MPPNRegressor

            model = MPPNRegressor(config.model_config, num_perspectives=num_perspectives) # type: ignore
            lightning_model = (
                None
                if lightning_datamodule is not None
                else MPPNLightningModule(config, model=model)
            )
            return model, lightning_model

        case _:
            raise ValueError(f"Unsupported model type: {model_type}")
