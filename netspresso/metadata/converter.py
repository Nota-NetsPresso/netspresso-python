from dataclasses import dataclass, field
from typing import List

from netspresso.clients.launcher.v2.schemas import ModelBase, ModelOption
from netspresso.clients.launcher.v2.schemas.task.convert.response_body import (
    ConvertTask,
)
from netspresso.metadata.common import ModelInfo, AvailableOptions
from netspresso.enums import (
    Status,
    TaskType,
    DataType,
    Framework,
    DeviceName,
    SoftwareVersion,
)


@dataclass
class ConvertInfo:
    convert_task_uuid: str = ""
    framework: Framework = ""
    display_framework: str = ""
    device_name: DeviceName = ""
    display_device_name: str = ""
    display_brand_name: str = ""
    data_type: DataType = ""
    software_version: SoftwareVersion = ""
    display_software_version: str = ""
    model_file_name: str = ""
    input_model_uuid: str = ""
    output_model_uuid: str = ""


@dataclass
class ConverterMetadata:
    status: Status = Status.IN_PROGRESS
    task_type: TaskType = TaskType.CONVERT
    input_model_path: str = ""
    converted_model_path: str = ""
    model_info: ModelInfo = field(default_factory=ModelInfo)
    convert_task_info: ConvertInfo = field(default_factory=ConvertInfo)
    available_options: List[AvailableOptions] = field(
        default_factory=lambda: [AvailableOptions()]
    )


class ConvertMetadataBuilder:
    def __init__(self):
        self.convert_metadata = ConverterMetadata()

    def set_input_model_path(self, input_model_path: str):
        self.convert_metadata.input_model_path = input_model_path
        return self

    def set_converted_model_path(self, converted_model_path: str):
        self.convert_metadata.converted_model_path = converted_model_path
        return self

    def set_model_info(self, model_base: ModelBase):
        self.convert_metadata.model_info.set()
        return self

    def set_convert_task_info(self, convert_task: ConvertTask):
        return self

    def set_available_options(self, model_options: List[ModelOption]):
        return self

    def build(self) -> ConverterMetadata:
        return self.convert_metadata
