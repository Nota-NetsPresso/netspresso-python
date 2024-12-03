import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Union

from netspresso.clients.launcher.v2.schemas import InputLayer
from netspresso.enums import QuantizationMode, QuantizationPrecision, SimilarityMetric


@dataclass
class QuantizationOptions:
    use_cuda: bool = False
    suggestion_only: bool = False
    metric: SimilarityMetric = SimilarityMetric.SNR  # Quantization quality metrics
    threshold: Union[float, int] = 0  # Quantization quality threshold
    weight_precision: QuantizationPrecision = QuantizationPrecision.INT8  # Weight precision
    activation_precision: QuantizationPrecision = QuantizationPrecision.INT8  # Activation precision

    def __post_init__(self):
        # weight_precision validation
        if self.weight_precision not in [QuantizationPrecision.INT8, QuantizationPrecision.FLOAT32]:
            if self.weight_precision == QuantizationPrecision.FLOAT16:
                raise ValueError("weight_precision FLOAT16 is only available for custom precision quantization.")
            raise ValueError(f"weight_precision must be either INT8 or FLOAT32, got {self.weight_precision}")

        # activation_precision validation
        if self.activation_precision not in [QuantizationPrecision.INT8, QuantizationPrecision.FLOAT32]:
            if self.activation_precision == QuantizationPrecision.FLOAT16:
                raise ValueError("activation_precision FLOAT16 is only available for custom precision quantization.")
            raise ValueError(f"activation_precision must be either INT8 or FLOAT32, got {self.activation_precision}")


@dataclass
class PlainQuantizationOption(QuantizationOptions):
    pass


@dataclass
class RecommendationOption(QuantizationOptions):
    suggestion_only: bool = True


@dataclass
class CustomQuantizeOption:
    use_cuda: bool = False
    metric: SimilarityMetric = SimilarityMetric.SNR  # Quantization quality metrics
    custom_precision: Dict = field(default_factory=dict)  # Custom precision by layer or operator
    weight_precision: QuantizationPrecision = QuantizationPrecision.INT8  # Weight precision
    activation_precision: QuantizationPrecision = QuantizationPrecision.INT8  # Activation precision


@dataclass
class AutomaticQuantizeOption(QuantizationOptions):
    use_cuda: bool = False
    suggestion_only: bool = False
    metric: SimilarityMetric = SimilarityMetric.SNR  # Quantization quality metrics
    threshold: Union[float, int] = 0  # Quantization quality threshold
    weight_precision: QuantizationPrecision = QuantizationPrecision.INT8  # Weight precision
    activation_precision: QuantizationPrecision = QuantizationPrecision.INT8  # Activation precision


@dataclass
class RequestQuantizeTask:
    input_model_id: str
    quantization_mode: QuantizationMode
    quantization_options: Union[QuantizationOptions, CustomQuantizeOption]
    input_layers: Optional[List[InputLayer]] = None

    def __post_init__(self):
        self.quantization_options = json.dumps(asdict(self.quantization_options))

        if self.input_layers:
            self.input_layers = json.dumps([asdict(input_layer) for input_layer in self.input_layers])
