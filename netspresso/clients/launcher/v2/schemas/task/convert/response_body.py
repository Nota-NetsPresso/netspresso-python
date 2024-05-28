import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional

from netspresso.clients.launcher.v2.schemas import (
    InputLayer,
    ModelOption,
    ResponseItem,
    ResponseItems,
    TaskInfo,
    TaskOption,
)
from netspresso.clients.launcher.v2.schemas.task.common import TaskStatusInfo
from netspresso.enums import Framework, TaskStatusForDisplay
from netspresso.metadata.converter import ConvertInfo


@dataclass
class ConvertTask:
    convert_task_id: str
    input_model_id: str
    output_model_id: str
    input_layer: InputLayer = field(default_factory=InputLayer)
    status: TaskStatusForDisplay = ""
    convert_task_option: Optional[TaskOption] = field(default_factory=TaskOption)

    def __init__(self, **kwargs):
        names = {f.name for f in dataclasses.fields(self)}
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)

        self.input_layer = InputLayer(**self.input_layer)
        self.convert_task_option = TaskOption(**self.convert_task_option)

    def to(self, model_file_name: str) -> ConvertInfo:
        device_info = self.convert_task_option.target_device

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

        convert_info.data_type = device_info.data_type

        convert_info.software_version = device_info.software_version
        convert_info.display_software_version = device_info.display_software_version

        return convert_info


@dataclass
class ResponseConvertTaskItem(ResponseItem):
    data: Optional[ConvertTask] = field(default_factory=dict)

    def __post_init__(self):
        self.data = ConvertTask(**self.data)


@dataclass
class ConvertOption:
    option_name: str
    display_option: str
    framework: Framework
    device: TaskInfo

    def __init__(self, **kwargs):
        names = {f.name for f in dataclasses.fields(self)}
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)

        self.device = TaskInfo(**self.device)


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


@dataclass
class DownloadModelUrl:
    ai_model_id: str
    presigned_download_url: str


@dataclass
class ResponseConvertDownloadModelUrlItem(ResponseItem):
    data: Optional[DownloadModelUrl] = field(default_factory=dict)

    def __post_init__(self):
        self.data = DownloadModelUrl(**self.data)


@dataclass
class ResponseConvertFrameworkOptionItems(ResponseItems):
    data: List[Optional[ModelOption]] = field(default_factory=list)

    def __post_init__(self):
        self.data = [ModelOption(**item) for item in self.data]
