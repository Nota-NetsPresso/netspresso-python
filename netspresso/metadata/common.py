import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from netspresso.enums.device import DeviceName, HardwareType, SoftwareVersion
from netspresso.enums.metadata import Status
from netspresso.enums.model import DataType, Framework


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
class LinkInfo:
    type: str
    value: str


@dataclass
class AdditionalData:
    origin: Optional[str] = ""
    error_log: Optional[str] = ""
    link: Optional[LinkInfo] = None


@dataclass
class ExceptionDetail:
    data: Optional[AdditionalData] = field(default_factory=AdditionalData)
    error_code: Optional[str] = ""
    name: Optional[str] = ""
    message: Optional[str] = ""


@dataclass
class BaseMetadata:
    status: Status = Status.IN_PROGRESS
    error_detail: ExceptionDetail = field(default_factory=ExceptionDetail)

    def asdict(self) -> Dict:
        _dict = json.loads(json.dumps(asdict(self)))
        return _dict

    def update_message(self, exception_detail):
        if isinstance(exception_detail, str):
            self.error_detail.message = exception_detail
        else:
            self.error_detail = ExceptionDetail(**exception_detail)

    def update_status(self, status: Status):
        self.status = status
