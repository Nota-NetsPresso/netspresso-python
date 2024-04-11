from enum import Enum
from typing import List, Optional, Union
from dataclasses import dataclass, field

from netspresso.enums import (
    DataType,
    HardwareType,
    Framework,
    SoftwareVersion,
    DisplaySoftwareVersion,
)


class Order(str, Enum):
    """ """

    DESC = "desc"
    ASC = "asc"


@dataclass(init=False)
class AuthorizationHeader:
    Authorization: str

    def __init__(self, access_token):
        self.Authorization = f"Bearer {access_token}"


@dataclass(init=False)
class UploadFile:
    files: List

    def __init__(self, file_name, file_content):
        self.files = [("file", (file_name, file_content))]


@dataclass
class RequestPagination:
    """ """

    start: int = 0
    size: int = 10
    order: Order = Order.DESC.value
    paging: bool = True


@dataclass
class ResponseItem:
    """ """

    data: Optional[object] = field(default_factory=dict)


@dataclass
class ResponseItems:
    """ """

    data: List[Optional[object]] = field(default_factory=list)


@dataclass
class ResponsePaginationItems:
    """ """

    result_count: int
    total_count: int
    data: List[Optional[object]] = field(default_factory=list)


@dataclass
class SoftwareVersionInfo:
    """ """

    software_version: Optional[Union[None, SoftwareVersion]] = None
    display_software_version: Optional[Union[None, DisplaySoftwareVersion]] = None


@dataclass
class DeviceInfo:
    """ """

    device_name: str
    display_device_name: str
    display_brand_name: str
    software_versions: Optional[List[SoftwareVersionInfo]] = field(default_factory=list)
    data_types: Optional[List[DataType]] = field(default_factory=list)
    hardware_types: Optional[List[HardwareType]] = field(default_factory=list)


@dataclass
class ModelOption:
    """ """

    framework: Optional[Framework] = ""
    display_framework: Optional[str] = ""
    devices: List[DeviceInfo] = field(default_factory=list)
