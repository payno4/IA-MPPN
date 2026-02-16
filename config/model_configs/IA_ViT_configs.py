from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json#(undefined=Undefined.RAISE)
@dataclass
class IA_VisionTransformerConfig():
    patches: dict = field(default_factory=lambda: {
        "size": (3, 3)
    })
    hidden_size: int = 256 

    mlp_dim: int = 1024
    num_heads: int = 8 # original 12
    num_layers: int = 6 # original 12
    attention_dropout_rate: float = 0.05
    dropout_rate: float = 0.05

    def __post_init__(self):
        assert self.hidden_size % self.num_heads == 0, \
            f"hidden_size {self.hidden_size} must be divisible by num_heads {self.num_heads}."
        
