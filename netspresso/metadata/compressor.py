from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from netspresso.enums.metadata import TaskType
from netspresso.enums.model import DataType, Framework
from netspresso.metadata.common import AvailableOption, BaseMetadata, InputShape
from netspresso.metadata.trainer import TrainingInfo


@dataclass
class ModelInfo:
    task: str = ""
    model: str = ""
    dataset: str = ""
    data_type: DataType = DataType.FP32
    framework: Framework = ""
    input_shapes: List[InputShape] = field(default_factory=list)


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
class CompressorMetadata(BaseMetadata):
    task_type: TaskType = TaskType.COMPRESS
    input_model_path: str = ""
    compressed_model_path: str = ""
    compressed_onnx_model_path: str = ""
    is_retrainable: bool = False
    model_info: ModelInfo = field(default_factory=ModelInfo)
    training_info: TrainingInfo = field(default_factory=TrainingInfo)
    compression_info: CompressionInfo = field(default_factory=CompressionInfo)
    results: Results = field(default_factory=Results)
    available_options: List[AvailableOption] = field(default_factory=list)

    def update_is_retrainable(self, is_retrainable):
        self.is_retrainable = is_retrainable

    def update_model_info(self, framework, input_shapes):
        self.model_info.framework = framework
        self.model_info.input_shapes = input_shapes

    def update_model_info_for_trainer(self, task, model, dataset):
        self.model_info.task = task
        self.model_info.model = model
        self.model_info.dataset = dataset

    def update_training_info(self, epochs, batch_size, learning_rate, optimizer):
        self.training_info.epochs = epochs
        self.training_info.batch_size = batch_size
        self.training_info.learning_rate = learning_rate
        self.training_info.optimizer = optimizer

    def update_compression_info(self, method, options, layers, ratio=0.0):
        self.compression_info.method = method
        self.compression_info.ratio = ratio
        self.compression_info.options = options
        self.compression_info.layers = layers

    def update_input_model_path(self, input_model_path):
        self.input_model_path = input_model_path

    def update_compressed_model_path(self, compressed_model_path):
        self.compressed_model_path = compressed_model_path

    def update_compressed_onnx_model_path(self, compressed_onnx_model_path):
        self.compressed_onnx_model_path = compressed_onnx_model_path

    def update_results(self, model, compressed_model):
        def update_model_fields(target, source):
            target.size = source.file_size_in_mb
            target.flops = source.detail.flops
            target.number_of_parameters = source.detail.trainable_parameters + source.detail.non_trainable_parameters
            target.trainable_parameters = source.detail.trainable_parameters
            target.non_trainable_parameters = source.detail.non_trainable_parameters
            target.number_of_layers = source.detail.number_of_layers if source.detail.number_of_layers != 0 else None
            target.model_id = source.ai_model_id

        update_model_fields(self.results.original_model, model)
        update_model_fields(self.results.compressed_model, compressed_model)

    def update_available_options(self, available_options):
        self.available_options = [available_option.to() for available_option in available_options]
