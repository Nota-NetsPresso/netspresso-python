from dataclasses import dataclass
from typing import Dict, List, Optional

from netspresso.clients.compressor.v2.schemas.common import ResponseItem, ResponsePaginationItems
from netspresso.enums.compression import CompressionMethod, GroupPolicy, LayerNorm, Policy, RecommendationMethod, StepOp


@dataclass
class ResponseCompression:
    compression_id: str
    compression_method: CompressionMethod
    is_completed: bool
    is_deleted: bool
    input_model_id: str
    original_model_id: str
    user_id: str
    available_layers: List
    parameters: Dict


@dataclass
class ResponseRecommendation:
    recommendation_id: str
    recommendation_method: RecommendationMethod
    recommendation_ratio: float
    available_layers: List
    parameters: Dict
    compression_id: str


@dataclass
class ResponseAvailableLayers:
    compression_method: CompressionMethod
    available_layers: List


@dataclass
class ResponseCompressionItem(ResponseItem):
    data: Optional[ResponseCompression] = None


@dataclass
class ResponseCompressionItems(ResponsePaginationItems):
    data: List[ResponseCompression]


@dataclass
class ResponseRecommendationItem(ResponseItem):
    data: Optional[ResponseRecommendation] = None


@dataclass
class ResponseRecommendationItems(ResponsePaginationItems):
    data: List[ResponseRecommendation]


@dataclass
class ResponseAvailableLayersItem(ResponseItem):
    data: Optional[ResponseAvailableLayers] = None
