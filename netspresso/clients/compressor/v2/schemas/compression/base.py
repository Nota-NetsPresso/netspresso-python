from dataclasses import dataclass, field
from typing import List

from netspresso.enums.compression import GroupPolicy, LayerNorm, Policy, StepOp


@dataclass
class OptionsBase:
    reshape_channel_axis: int = -1

    def __post_init__(self):
        valid_values = [0, 1, -1, -2]
        if self.reshape_channel_axis not in valid_values:
            raise ValueError(
                f"The reshape_channel_axis value is in the range [0, 1, -1, -2], but got {self.reshape_channel_axis}"
            )


@dataclass
class Options(OptionsBase):
    policy: Policy = Policy.AVERAGE
    layer_norm: LayerNorm = LayerNorm.STANDARD_SCORE
    group_policy: GroupPolicy = GroupPolicy.AVERAGE
    step_size: int = 2
    step_op: StepOp = StepOp.ROUND
    reverse: bool = False


@dataclass
class RecommendationOptions(Options):
    min_num_of_value: int = 8


@dataclass
class Layer:
    use: bool = False
    name: str = "input"
    channels: List[int] = field(default_factory=list)
    values: List[int] = field(default_factory=list)
