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
from netspresso.metadata.converter import ConvertInfo


@dataclass
class ConvertTask:
    convert_task_id: str
    input_model_id: str
    output_model_id: str
    input_layer: InputLayer = field(default_factory=InputLayer)
    status: TaskStatusForDisplay = ""
    convert_task_option: Optional[ModelOption] = field(default_factory=ModelOption)

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)

        self.input_layer = InputLayer(**self.input_layer)
        self.convert_task_option = ModelOption(**self.convert_task_option)

    def to(self, model_file_name: str) -> ConvertInfo:
        device_info = self.convert_task_option.devices[0]

        convert_info = ConvertInfo()
        convert_info.convert_task_uuid = self.convert_task_id
        convert_info.framework = self.convert_task_option.framework
        convert_info.display_framework = self.convert_task_option.display_framework
        convert_info.input_model_uuid = self.input_model_id
        convert_info.output_model_uuid = self.output_model_id
        convert_info.model_file_name = model_file_name

        convert_info.device_name = device_info.device_name
        convert_info.display_device_name = device_info.display_device_name
        convert_info.display_brand_name = device_info.display_brand_name

        convert_info.data_type = device_info.data_types[0]

        convert_info.software_version = device_info.software_versions[
            0
        ].software_version
        convert_info.display_software_version = device_info.software_versions[
            0
        ].display_software_version

        return convert_info


@dataclass
class ResponseConvertTaskItem(ResponseItem):
    data: Optional[ConvertTask] = field(default_factory=dict)

    def __post_init__(self):
        self.data = ConvertTask(**self.data)


@dataclass
class ConvertOption:
    option_name: str
    framework: str
    device: DeviceInfo

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)

        self.device = DeviceInfo(**self.device)


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
