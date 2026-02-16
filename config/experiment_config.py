
from dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ExperimentConfig:
    model_name: str = "IA_ViTransformer"  # ExPeRT, IA_ViTransformer, resnet
    experiment_name: str = "default_experiment"
    run_name: str = "default_run"
    description: str = "This is a default experiment configuration."
    device: str = "cpu"
    seed: int = 42
    console_messages: bool = True
