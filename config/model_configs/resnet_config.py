from dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json#(undefined=Undefined.RAISE)
@dataclass
class ResNetConfig():
    model_name: str = "resnet"