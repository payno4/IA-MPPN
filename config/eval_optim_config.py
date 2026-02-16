from dataclasses import dataclass, field

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class EvaluationConfig:
    model_path_1: str = ""
    model_path_2: str = ""
    model_type: str = ""
    patch_size: int = 9 # for RegionPerturbation, has to match patch_size in model

@dataclass_json
@dataclass
class OptimConfig:
    study_name: str = "Hyperparameter Optimization "
    results_file_name: str = "optimization_study"
    model_type: str = "MPPN"
    n_trials: int = 100
    max_train_epochs: int = 200

@dataclass_json
@dataclass
class EvalOptimConfig:
    optim: OptimConfig = field(default_factory=OptimConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)