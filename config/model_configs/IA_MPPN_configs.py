from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class PerspectiveConfigs():
    patches: dict = field(default_factory=lambda: {"size": (3, 3)})
    hidden_size: int = 256
    mlp_dim: int = 1024
    num_heads: int = 8
    num_layers: int = 6
    attention_dropout_rate: float = 0.05
    dropout_rate: float = 0.05

@dataclass_json
@dataclass
class RelationsConfig():
    hidden_size: int = 256
    num_hidden_layers: int = 6
    num_attention_heads: int = 8
    mlp_dim: int = 1024
    attention_dropout_rate: float = 0.05
    dropout_rate: float = 0.05

@dataclass_json
@dataclass
class IA_MPPNConfig():
    perspective_model_config: PerspectiveConfigs = field(default_factory=PerspectiveConfigs)
    relation_model_config: RelationsConfig = field(default_factory=RelationsConfig)
    ablation_mode: int = 0  # 0: full model, 1: only perspectives with averaging, 2: perspectives with flattener, 3: perspectives + relation without flattener