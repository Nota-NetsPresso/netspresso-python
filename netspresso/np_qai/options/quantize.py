from dataclasses import dataclass
from enum import Enum
from typing import Optional

from qai_hub import QuantizeDtype


class RangeScheme(str, Enum):
    AUTO = "auto"
    MSE_MINIMIZER = "mse_minimizer"
    MIN_MAX = "min_max"


class QuantizePrecision:
    INT8 = QuantizeDtype.INT8
    INT16 = QuantizeDtype.INT16
    INT4 = QuantizeDtype.INT4


@dataclass
class QuantizeOptions:
    """
    Quantize options for the model.

    Note:
        For details, see `QuantizeOptions in QAI Hub API <https://app.aihub.qualcomm.com/docs/hub/api.html#quantize-options>`_.
    """

    range_scheme: Optional[RangeScheme] = RangeScheme.AUTO

    def to_cli_string(self) -> str:
        args = []
        if self.range_scheme is not None:
            args.append(f"--range_scheme {self.range_scheme}")

        return " ".join(args)
