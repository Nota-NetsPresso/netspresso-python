from dataclasses import dataclass, field
from typing import Union

from netspresso.enums import (
    QuantizationDataType,
    QuantizationMode,
    SimilarityMetric,
    TaskType,
)
from netspresso.metadata.common import BaseMetadata, ModelInfo


@dataclass
class QuantizeInfo:
    quantize_task_uuid: str = ""
    model_file_name: str = ""
    quantization_mode: QuantizationMode = QuantizationMode.PLAIN_QUANTIZATION
    metric: SimilarityMetric = SimilarityMetric.SNR
    threshold: Union[float, int] = 0
    weight_quantization_bandwidth: QuantizationDataType = QuantizationDataType.INT8
    activation_quantization_bandwidth: QuantizationDataType = QuantizationDataType.INT8
    input_model_uuid: str = ""
    output_model_uuid: str = ""


@dataclass
class QuantizerMetadata(BaseMetadata):
    task_type: TaskType = TaskType.QUANTIZE
    input_model_path: str = ""
    quantized_model_path: str = ""
    model_info: ModelInfo = field(default_factory=ModelInfo)
    quantize_task_info: QuantizeInfo = field(default_factory=QuantizeInfo)
