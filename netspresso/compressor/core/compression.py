from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class AvailableLayer:
    """Represents an available layer for compression.

    Attributes:
        name (str): The name of the layer.
        values (List[Any]): The compression parameters for the layer.
        use (bool): The compression selection status for the layer.
        channels (List[int]): The channel information for the layer.
    """

    name: str
    values: List[Any] = field(default_factory=list)
    use: bool = False
    channels: List[int] = field(default_factory=list)


@dataclass
class CompressionInfo:
    """Represents compression information for a model.

    Attributes:
        compressed_model_id (str): The ID of the compressed model.
        compression_id (str): The ID of the compression.
        compression_method (str): The compression method used.
        available_layers (List[AvailableLayer]): The compressible layers information.

            AvailableLayer Attributes:
                - name (str): The name of the layer.
                - values (List[Any]): The compression parameters for the layer.
                - use (bool): The compression selection status for the layer.
                - channels (List[int]): The channel information for the layer.

        original_model_id (str): The ID of the original model.
    """

    compressed_model_id: str = ""
    compression_id: str = ""
    compression_method: str = ""
    available_layers: List[AvailableLayer] = field(default_factory=list)
    original_model_id: str = ""
