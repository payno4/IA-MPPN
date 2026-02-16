
from dataclasses import dataclass
from pathlib import Path

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class IOConfig:
    static_dir: str = "./static/"
    optimizer_dir: str = static_dir +"/optimizer_states/"
    mlflow_dir: str = static_dir +"/mlflow/"
    checkpoints_dir: str = static_dir +"/checkpoints/"
    h5_dir: str = static_dir +"/h5_files/"
    raw_data_dir: str = static_dir +"/raw_data/"
    analysis_dir: str = static_dir +"/analysis/"

    def __post_init__(self):
        self.static_dir:Path = Path(self.static_dir)
        self.optimizer_dir:Path = Path(self.optimizer_dir)
        self.mlflow_dir:Path = Path(self.mlflow_dir)
        self.checkpoints_dir:Path = Path(self.checkpoints_dir)
        self.h5_dir:Path = Path(self.h5_dir)
        self.raw_data_dir:Path = Path(self.raw_data_dir)
        self.analysis_dir:Path = Path(self.analysis_dir)
        
        self.static_dir.mkdir(parents=True, exist_ok=True)
        self.optimizer_dir.mkdir(parents=True, exist_ok=True)
        self.mlflow_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.h5_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)