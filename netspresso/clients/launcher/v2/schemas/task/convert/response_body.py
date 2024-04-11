import dataclasses
from dataclasses import dataclass, field
from typing import Optional, List

from netspresso.enums import TaskStatusForDisplay
from netspresso.clients.launcher.v2.schemas.task.common import TaskStatusInfo
from netspresso.clients.launcher.v2.schemas import (
    DeviceInfo,
    InputLayer,
    ModelOption,
    ResponseItem,
    ResponseItems,
)


@dataclass
class ConvertTask:
    convert_task_id: str
    input_model_id: str
    output_model_id: str
    input_layer: InputLayer
    status: TaskStatusForDisplay
    convert_task_option: Optional[ModelOption] = None

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)


@dataclass
class ResponseConvertTaskItem(ResponseItem):
    data: Optional[ConvertTask] = field(default_factory=dict)

    def __post_init__(self):
        self.data = ConvertTask(**self.data)


@dataclass(init=False)
class ConvertOption:
    option_name: str
    framework: str
    device: DeviceInfo

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)


@dataclass
class ResponseConvertOptionItems(ResponseItems):
    data: List[Optional[ConvertOption]] = field(default_factory=list)

    def __post_init__(self):
        self.data = [ConvertOption(**item) for item in self.data]


@dataclass
class ResponseConvertStatusItem(ResponseItem):
    data: TaskStatusInfo = field(default_factory=TaskStatusInfo)

    def __post_init__(self):
        self.data = TaskStatusInfo(**self.data)
