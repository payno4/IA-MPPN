from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class MPPNConfig():
    feature_size: int = 64
    representation_dim: int = 128