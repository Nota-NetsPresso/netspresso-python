from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Union

from netspresso.enums.device import DeviceName, HardwareType, SoftwareVersion
from netspresso.enums.model import (
    DataType,
    Framework,
)


@dataclass
class InputShape:
    batch: int = 1
    channel: int = 3
    dimension: List[int] = field(default_factory=list)


@dataclass
class ModelInfo:
    data_type: DataType = ""
    framework: Framework = ""
    input_shapes: List[InputShape] = field(default_factory=list)


@dataclass
class SoftwareVersions:
    software_version: SoftwareVersion = ""
    display_software_versions: str = ""


@dataclass
class DeviceInfo:
    device_name: DeviceName = ""
    display_device_name: str = ""
    display_brand_name: str = ""
    software_versions: List[SoftwareVersions] = field(default_factory=list)
    data_types: List[DataType] = field(default_factory=list)
    hardware_types: List[HardwareType] = field(default_factory=list)


@dataclass
class AvailableOption:
    framework: Framework = ""
    display_framework: str = ""
    devices: List[DeviceInfo] = field(default_factory=list)



@dataclass
class ErrorFormat:
    raw_message: Union[str, Dict] = field(repr=False, compare=False)
    message: Optional[str] = ""
    error_log: Optional[str] = ""

    def __post_init__(self):
        if isinstance(self.raw_message, str):
            self.message = self.raw_message
        else:
            self.message = self.raw_message["message"]
            self.error_log = self.raw_message["data"]["error_log"]

    def asdict(self) -> Dict:
        _dict = {k: v for k, v in asdict(self).items() if k != 'raw_message'}
        return _dict
