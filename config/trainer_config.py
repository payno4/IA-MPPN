from dataclasses import dataclass

from dataclasses_json import dataclass_json


@dataclass_json#(undefined=Undefined.RAISE)
@dataclass
class TrainerConfig:
    epochs: int = 150
    lr: float = 5e-5
    weight_decay: float = 1e-4
    scheduler: str = "onecycle" #onecycle, cosine_warmup, none
    warmup_ratio: float = 0.1
    optimizer: str = "adamw" #adam, adamw, sgd
    criterion: str = "MSELoss" #MSELoss, L1Loss, SmoothL1Loss
    beta1: float = 0.05 #especially for SmoothL1Loss
    beta2: float = 0.999 #not used yet