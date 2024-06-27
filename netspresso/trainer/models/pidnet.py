from dataclasses import dataclass, field
from typing import Any, Dict, List

from netspresso.trainer.models.base import ArchitectureConfig, ModelConfig


@dataclass
class PIDNetArchitectureConfig(ArchitectureConfig):
    full: Dict[str, Any] = field(
        default_factory=lambda: {
            "name": "pidnet",
            "m": 2,
            "n": 3,
            "channels": 32,
            "ppm_channels": 96,
            "head_channels": 128,
        }
    )


@dataclass
class PIDNetModelConfig(ModelConfig):
    task: str = "segmentation"
    name: str = "pidnet_s"
    architecture: ArchitectureConfig = field(default_factory=lambda: PIDNetArchitectureConfig())
    losses: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {"criterion": "pidnet_loss", "ignore_index": 255, "weight": None},
        ]
    )
