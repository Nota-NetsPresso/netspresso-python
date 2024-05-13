from dataclasses import dataclass, field
from typing import List

from netspresso.enums.compression import CompressionMethod, GroupPolicy, LayerNorm, Policy, RecommendationMethod, StepOp


@dataclass
class OptionsBase:
    reshape_channel_axis: int = -1

    def __post_init__(self):
        valid_values = [0, 1, -1, -2]
        if self.reshape_channel_axis not in valid_values:
            raise ValueError(
                f"The reshape_channel_axis value is in the range [0, 1, -1, -2], but got {self.reshape_channel_axis}"
            )


class Options(OptionsBase):
    policy: Policy = Policy.AVERAGE
    layer_norm: LayerNorm = LayerNorm.STANDARD_SCORE
    group_policy: GroupPolicy = GroupPolicy.AVERAGE
    step_size: int = 2
    step_op: StepOp = StepOp.ROUND
    reverse: bool = False


class RecommendationOptions(Options):
    min_num_of_value: int = 8


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
        if self.recommendation_method in ["slamp"]:
            assert 0 < self.recommendation_ratio <= 1, "The ratio range for SLAMP is 0 < ratio < = 1."
        elif self.recommendation_method in ["vbmf"]:
            assert -1 <= self.recommendation_ratio <= 1, "The ratio range for VBMF is -1 <= ratio <= 1."


@dataclass
class RequestUpdateCompression:
    available_layers: List
    options: Options = field(default_factory=Options)

    def __post_init__(self):
        if all(not available_layer["values"] for available_layer in self.available_layers):
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
