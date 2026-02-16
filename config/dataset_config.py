
from dataclasses import dataclass, field
from math import isclose

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class DatasetConfig:
    h5_file_name: str = "default_dataset"
    split_idxs: tuple[float, float, float] = field(
        default_factory=lambda: (0.7, 0.25, 0.05) # train, val, test
    ) 
    trace_duration_conversion: str = "hours" #options: seconds, minutes, hours, days

    def __post_init__(self):
        assert self.trace_duration_conversion in ["seconds", "minutes", "hours", "days"], f"trace_duration_conversion must be one of 'seconds', 'minutes', 'hours', 'days'. Now it is {self.trace_duration_conversion}."
        if len(self.split_idxs) != 3:
            raise ValueError("split_idxs must be a tuple of three integers (train, val, test).")
        if not all(isinstance(idx, (int, float)) for idx in self.split_idxs):
            raise TypeError("All elements in split_idxs must be integers or floats.")
        self.split_idxs = (self.split_idxs[0], self.split_idxs[1], self.split_idxs[2])
        if not isclose(sum(self.split_idxs), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"The sum of split_idxs must be 1.0. Now it is {sum(self.split_idxs)}.")
        if not all(0.0 <= idx <= 1.0 for idx in self.split_idxs):
            raise ValueError("All split_idxs must be between 0.0 and 1.0.")