import json
from dataclasses import dataclass, field, asdict
from typing import Optional

from netspresso.clients.launcher.v2.schemas import InputLayer
from netspresso.enums import DataType, DeviceName, Framework, SoftwareVersion


@dataclass
class RequestConvert:
    input_model_id: str
    target_framework: Framework
    target_device_name: DeviceName
    data_type: Optional[DataType] = None
    input_layer: Optional[str] = field(default_factory=InputLayer)
    software_version: Optional[SoftwareVersion] = ""

    def __post_init__(self):
        self.input_layer = json.dumps(asdict(self.input_layer))