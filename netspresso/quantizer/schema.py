from dataclasses import dataclass
from typing import List

from netspresso.enums import QuantizationPrecision


@dataclass
class PrecisionByLayer:
    name: str
    precision: QuantizationPrecision


@dataclass
class PrecisionByOperator:
    type: str
    precision: QuantizationPrecision


@dataclass
class RecommendationPrecisions:
    layers: List[PrecisionByLayer]
    operators: List[PrecisionByOperator]
