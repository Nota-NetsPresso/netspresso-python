from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class ComputeUnit(str, Enum):
    ALL = "all"
    NPU = "npu"
    GPU = "gpu"
    CPU = "cpu"


@dataclass
class CommonOptions:
    """
    Common options for all tasks.

    Args:
        compute_unit: Specifies the target compute unit(s)

    Note:
        For details, see `CommonOptions in QAI Hub API <https://app.aihub.qualcomm.com/docs/hub/api.html#common-options>`_.
    """

    compute_unit: Optional[List[ComputeUnit]] = None
