from dataclasses import dataclass, field
from typing import List

from netspresso.enums import (
    DataType,
    Framework,
    DeviceName,
    SoftwareVersion,
    HardwareType,
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
    input_shapes: List[InputShape] = field(default_factory=lambda: [InputShape()])


@dataclass
class SoftwareVersions:
    software_version: SoftwareVersion = ""
    display_software_versions: str = ""


@dataclass()
class DeviceInfo:
    device_name: DeviceName = ""
    display_device_name: str = ""
    display_brand_name: str = ""
    software_versions: SoftwareVersions = ""
    data_types: List[DataType] = field(default_factory=list)
    hardware_types: List[HardwareType] = field(default_factory=list)


@dataclass()
class AvailableOptions:
    framework: Framework = ""
    display_framework: str = ""
    devices: List[DeviceInfo] = field(default_factory=lambda: [DeviceInfo()])
