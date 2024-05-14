from dataclasses import dataclass, field
from typing import List, Optional

from netspresso.enums.model import Framework


@dataclass
class InputLayer:
    name: str = "input"
    batch: int = 1
    channel: int = 3
    dimension: List[int] = field(default_factory=list)


@dataclass
class RequestCreateModel:
    object_name: str


@dataclass
class RequestUploadModel:
    ai_model_id: str
    presigned_upload_url: str


@dataclass
class RequestValidateModel:
    display_name: Optional[str]
    framework: Framework = Framework.PYTORCH
    input_layers: List[InputLayer]

    def __post_init__(self):
        new_input_layers = [input_layer.model_dump() for input_layer in self.input_layers]
        if self.framework == Framework.PYTORCH and not new_input_layers:
            raise Exception()
        self.input_layers = new_input_layers
