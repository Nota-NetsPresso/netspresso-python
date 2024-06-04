from dataclasses import dataclass
from typing import Optional

from netspresso.metadata.common import InputShape, ModelInfo


@dataclass
class InputLayer:
    name: Optional[str] = None
    batch: Optional[int] = None
    channel: Optional[int] = None
    dimension: Optional[list] = None

    def to(self) -> InputShape:
        input_shape = InputShape()
        input_shape.batch = self.batch
        input_shape.channel = self.channel
        input_shape.dimension = self.dimension
        return input_shape


@dataclass
class ModelDetail:
    data_type: Optional[str] = None
    ai_model_format: Optional[str] = None
    framework: Optional[str] = None
    flops: Optional[float] = None
    non_trainable_parameters: Optional[int] = None
    trainable_parameters: Optional[int] = None
    number_of_layers: Optional[int] = None
    graph_info: Optional[dict] = None
    input_layer: Optional[InputLayer] = None

    def __post_init__(self):
        self.input_layer = InputLayer(**self.input_layer)


@dataclass
class ModelStatus:
    is_deleted: Optional[bool] = False
    is_convertible: Optional[bool] = False
    is_compressible: Optional[bool] = False
    is_benchmarkable: Optional[bool] = False
    is_uploaded: Optional[bool] = False


@dataclass
class ModelBase:
    user_id: str
    ai_model_id: str
    uploaded_file_name: Optional[str]
    file_size_in_mb: Optional[float]
    md5_checksum: Optional[str]
    bucket_name: Optional[str]
    object_path: Optional[str]
    task: str
    display_name: Optional[str]
    error_log: Optional[str]
    status: Optional[ModelStatus]
    detail: Optional[ModelDetail]

    def __post_init__(self):
        self.status = ModelStatus(**self.status)
        self.detail = ModelDetail(**self.detail)

    def to(self) -> ModelInfo:
        model_info = ModelInfo()
        model_info.data_type = self.detail.data_type
        model_info.framework = self.detail.framework
        model_info.input_shapes.append(self.detail.input_layer.to())
        return model_info
