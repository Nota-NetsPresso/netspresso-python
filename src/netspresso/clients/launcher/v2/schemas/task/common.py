from dataclasses import dataclass

from netspresso.enums import TaskStatusForDisplay


@dataclass
class Device:
    device_brand: str
    device_name: str


@dataclass
class TaskStatusInfo:
    status: TaskStatusForDisplay
