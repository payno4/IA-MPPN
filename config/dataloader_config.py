
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json, config

@dataclass_json
@dataclass
class t_dataloaderConfig:
    num_workers: int = 4
    batch_size: int = 32
    shuffle: bool = False
    pin_memory: bool = True
    pin_memory_device: str = "cuda"
    persistent_workers: bool = True
    drop_last: bool = False

@dataclass_json
@dataclass
class DataloaderConfig:
    h5_file_name: str = "default_dataset"
    dataset_name: str = "default_dataset"
    weighted_sampler: bool = False
    bins: int = 100
    transform: str = "none"  # none, normalize, standardize
    train_dataloader: t_dataloaderConfig = field(default_factory=t_dataloaderConfig)
    test_dataloader: t_dataloaderConfig = field(default_factory=t_dataloaderConfig)
    validation_dataloader: t_dataloaderConfig = field(default_factory=t_dataloaderConfig)
    target_min: float = field(
        default=0,
        metadata=config(exclude=lambda _: True)
    )
    target_max: float = field(
        default=0,
        metadata=config(exclude=lambda _: True)
    )
    max_len_log: int = field(
        default=0,
        metadata=config(exclude=lambda _: True)
    )
    input_shape: list[int] = field(
        default_factory=lambda: [0, 0, 0],
        metadata=config(exclude=lambda _: True)
    ) # [H, W, C]

