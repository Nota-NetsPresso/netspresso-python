import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Union

from netspresso.clients.launcher.v2.schemas import InputLayer
from netspresso.enums import QuantizationDataType, QuantizationMode, SimilarityMetric


@dataclass
class QuantizationOptions:
    metric: SimilarityMetric = field(default=SimilarityMetric.SNR)
    threshold: Union[float, int] = field(default=0)
    weight_quantization_bandwidth: QuantizationDataType = field(default=QuantizationDataType.INT8)
    activation_quantization_bandwidth: QuantizationDataType = field(default=QuantizationDataType.INT8)


@dataclass
class RequestQuantize:
    input_model_id: str
    quantization_mode: QuantizationMode = field(default=QuantizationMode.PLAIN_QUANTIZATION)
    quantization_options: QuantizationOptions = field(default_factory=QuantizationOptions)
    input_layers: Optional[List[InputLayer]] = None

    def __post_init__(self):
        self.quantization_options = json.dumps(asdict(self.quantization_options))

        if self.input_layers:
            self.input_layers = json.dumps([asdict(input_layer) for input_layer in self.input_layers])
