import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from netspresso.clients.compressor.v2.schemas.common import ResponseItem, ResponsePaginationItems
from netspresso.clients.compressor.v2.schemas.compression.base import Layer, Options
from netspresso.enums.compression import CompressionMethod, RecommendationMethod


@dataclass
class ResponseCompression:
    compression_id: str
    compression_method: CompressionMethod
    is_completed: bool
    is_deleted: bool
    input_model_id: str
    original_model_id: str
    user_id: str
    available_layers: List[Layer]
    options: Dict

    def __init__(self, **kwargs):
        names = {f.name for f in dataclasses.fields(self)}
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)

        self.available_layers = [Layer(**layer) for layer in self.available_layers]


@dataclass
class ResponseRecommendation:
    recommendation_id: str
    recommendation_method: RecommendationMethod
    recommendation_ratio: float
    available_layers: List[Layer]
    options: Dict
    compression_id: str

    def __init__(self, **kwargs):
        names = {f.name for f in dataclasses.fields(self)}
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)

        self.available_layers = [Layer(**layer) for layer in self.available_layers]


@dataclass
class ResponseSelectMethod:
    input_model_id: str
    compression_method: CompressionMethod
    options: Options
    available_layers: List[Layer]

    def __init__(self, **kwargs):
        names = {f.name for f in dataclasses.fields(self)}
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)

        self.available_layers = [Layer(**layer) for layer in self.available_layers]


@dataclass
class ResponseCompressionItem(ResponseItem):
    data: Optional[ResponseCompression] = field(default_factory=dict)

    def __post_init__(self):
        self.data = ResponseCompression(**self.data)


@dataclass
class ResponseCompressionItems(ResponsePaginationItems):
    data: List[ResponseCompression] = field(default_factory=list)

    def __post_init__(self):
        self.data = [ResponseCompression(**compression) for compression in self.data]


@dataclass
class ResponseRecommendationItem(ResponseItem):
    data: Optional[ResponseRecommendation] = field(default_factory=dict)

    def __post_init__(self):
        self.data = ResponseRecommendation(**self.data)


@dataclass
class ResponseSelectMethodItem(ResponseItem):
    data: Optional[ResponseSelectMethod] = None

    def __post_init__(self):
        self.data = ResponseSelectMethod(**self.data)
