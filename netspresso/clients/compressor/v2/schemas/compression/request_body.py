from dataclasses import dataclass, field
from typing import List

from netspresso.clients.compressor.v2.schemas.compression.base import Layer, Options, RecommendationOptions
from netspresso.enums.compression import CompressionMethod, RecommendationMethod


@dataclass
class RequestCreateCompression:
    ai_model_id: str
    compression_method: CompressionMethod
    options: Options = field(default_factory=Options)


@dataclass
class RequestCreateRecommendation:
    recommendation_method: RecommendationMethod
    recommendation_ratio: float
    options: RecommendationOptions = field(default_factory=RecommendationOptions)

    def __post_init__(self):
        if self.recommendation_method in [RecommendationMethod.SLAMP]:
            assert 0 < self.recommendation_ratio <= 1, "The ratio range for SLAMP is 0 < ratio < = 1."
        elif self.recommendation_method in [RecommendationMethod.VBMF]:
            assert -1 <= self.recommendation_ratio <= 1, "The ratio range for VBMF is -1 <= ratio <= 1."


@dataclass
class RequestUpdateCompression:
    available_layers: List[Layer]
    options: Options = field(default_factory=Options)

    def __post_init__(self):
        if all(not available_layer.values for available_layer in self.available_layers):
            raise Exception(
                "The available_layer.values all empty. please put in the available_layer.values to compress."
            )


@dataclass
class RequestAutomaticCompressionParams:
    compression_ratio: float = 0.5


@dataclass
class RequestAvailableLayers:
    compression_method: CompressionMethod
    options: Options = field(default_factory=Options)
