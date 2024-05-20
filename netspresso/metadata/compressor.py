import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from netspresso.enums.metadata import Status, TaskType

from netspresso.utils.metadata.default.common import TargetDevice
from netspresso.metadata.common import AvailableOptions, ModelInfo


@dataclass
class CompressionInfo:
    method: str = ""
    ratio: float = 0.0
    options: Dict[str, Any] = None
    layers: List[Dict] = field(default_factory=list)


@dataclass
class Model:
    size: int = 0
    flops: int = 0
    number_of_parameters: int = 0
    trainable_parameters: int = 0
    non_trainable_parameters: int = 0
    number_of_layers: Optional[int] = None
    model_id: str = ""


@dataclass
class Results:
    original_model: Model = field(default_factory=Model)
    compressed_model: Model = field(default_factory=Model)


@dataclass
class CompressorMetadata:
    status: Status = Status.IN_PROGRESS
    message: str = ""
    task_type: TaskType = TaskType.COMPRESS
    input_model_path: str = ""
    compressed_model_path: str = ""
    compressed_onnx_model_path: str = ""
    is_retrainable: bool = True
    model_info: ModelInfo = field(default_factory=ModelInfo)
    compression_info: CompressionInfo = field(default_factory=CompressionInfo)
    results: Results = field(default_factory=Results)
    available_options: List[AvailableOptions] = field(default_factory=list)

    def asdict(self) -> Dict:
        _dict = json.loads(json.dumps(asdict(self)))
        return _dict

    def update_status(self, status: Status):
        self.status = status

    def update_model_info(self, framework, input_shapes):
        self.model_info.framework = framework
        self.model_info.input_shapes = input_shapes

    def update_compression_info(self, method, options, layers, ratio=0.0):
        self.compression_info.method = method
        self.compression_info.ratio = ratio
        self.compression_info.options = options
        self.compression_info.layers = layers

    def update_compressed_model_path(self, compressed_model_path):
        self.compressed_model_path = compressed_model_path

    def update_compressed_onnx_model_path(self, compressed_onnx_model_path):
        self.compressed_onnx_model_path = compressed_onnx_model_path

    def update_results(self, model, compressed_model):
        def update_model_fields(target, source):
            target.size = source.file_size_in_mb
            target.flops = source.detail.flops
            target.number_of_parameters = source.detail.trainable_parameter + source.detail.non_trainable_parameter
            target.trainable_parameters = source.detail.trainable_parameter
            target.non_trainable_parameters = source.detail.non_trainable_parameter
            target.number_of_layers = source.detail.number_of_layers if source.detail.number_of_layers != 0 else None
            target.model_id = source.ai_model_id

        update_model_fields(self.results.original_model, model)
        update_model_fields(self.results.compressed_model, compressed_model)

    def update_available_options(self, available_options):
        # TODO
        pass