from dataclasses import dataclass, field

from netspresso.enums import TaskType
from netspresso.metadata.common import BaseMetadata, ModelInfo


@dataclass
class QuantizeInfo:
    quantize_task_uuid: str = ""
    model_file_name: str = ""
    input_model_uuid: str = ""
    output_model_uuid: str = ""


@dataclass
class QuantizerMetadata(BaseMetadata):
    task_type: TaskType = TaskType.QUANTIZE
    input_model_path: str = ""
    quantized_model_path: str = ""
    model_info: ModelInfo = field(default_factory=ModelInfo)
    quantize_info: QuantizeInfo = field(default_factory=QuantizeInfo)
