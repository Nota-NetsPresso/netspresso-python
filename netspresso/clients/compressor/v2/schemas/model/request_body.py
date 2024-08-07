from dataclasses import asdict, dataclass, field
from typing import List, Optional

from netspresso.enums.model import Framework
from netspresso.exceptions.compressor import NotFillInputLayersException


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
    url: str


@dataclass
class RequestValidateModel:
    input_layers: List[InputLayer]
    display_name: Optional[str] = ""
    framework: Framework = Framework.PYTORCH

    def __post_init__(self):
        new_input_layers = list(self.input_layers)
        if self.framework == Framework.PYTORCH and not new_input_layers:
            raise NotFillInputLayersException()
        self.input_layers = new_input_layers
