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
    compute_unit: Optional[List[ComputeUnit]] = None
