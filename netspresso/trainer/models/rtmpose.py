from dataclasses import dataclass, field
from typing import Any, Dict, List

from netspresso.trainer.models.base import ArchitectureConfig, ModelConfig
from netspresso.trainer.models.mobilenetv3 import MobileNetV3SmallArchitectureConfig


@dataclass
class PoseEstimationMobileNetV3SmallModelConfig(ModelConfig):
    task: str = "pose_estimation"
    name: str = "mobilenet_v3_small"
    architecture: ArchitectureConfig = field(
        default_factory=lambda: MobileNetV3SmallArchitectureConfig(
            head={
                "name": "rtmcc",
                "params": {
                    "conv_kernel": 7,
                    "attention_channels": 256,
                    "attention_act_type": "silu",
                    "attention_pos_enc": False,
                    "s": 128,
                    "expansion_factor": 2,
                    "dropout_rate": 0.0,
                    "drop_path": 0.0,
                    "use_rel_bias": False,
                    "simcc_split_ratio": 2.0,
                    "target_size": [256, 256],
                    "backbone_stride": 32,
                },
            }
        )
    )
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [{"criterion": "rtmcc_loss", "weight": None}])
