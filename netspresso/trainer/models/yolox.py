from dataclasses import dataclass, field
from typing import Any, Dict, List

from netspresso.trainer.models.base import ArchitectureConfig, CheckpointConfig, ModelConfig


@dataclass
class CSPDarkNetXArchitectureConfig(ArchitectureConfig):
    backbone: Dict[str, Any] = field(
        default_factory=lambda: {
            "name": "cspdarknet",
            "params": {
                "dep_mul": 1.33,
                "wid_mul": 1.25,
                "act_type": "silu",
            },
            "stage_params": None,
        }
    )


@dataclass
class CSPDarkNetLArchitectureConfig(ArchitectureConfig):
    backbone: Dict[str, Any] = field(
        default_factory=lambda: {
            "name": "cspdarknet",
            "params": {
                "dep_mul": 1.0,
                "wid_mul": 1.0,
                "act_type": "silu",
            },
            "stage_params": None,
        }
    )


@dataclass
class CSPDarkNetMArchitectureConfig(ArchitectureConfig):
    backbone: Dict[str, Any] = field(
        default_factory=lambda: {
            "name": "cspdarknet",
            "params": {
                "dep_mul": 0.67,
                "wid_mul": 0.75,
                "act_type": "silu",
            },
            "stage_params": None,
        }
    )


@dataclass
class CSPDarkNetSArchitectureConfig(ArchitectureConfig):
    backbone: Dict[str, Any] = field(
        default_factory=lambda: {
            "name": "cspdarknet",
            "params": {
                "dep_mul": 0.33,
                "wid_mul": 0.5,
                "act_type": "silu",
            },
            "stage_params": None,
        }
    )


@dataclass
class DetectionYoloXXModelConfig(ModelConfig):
    task: str = "detection"
    name: str = "yolox_x"
    architecture: ArchitectureConfig = field(
        default_factory=lambda: CSPDarkNetXArchitectureConfig(
            neck={
                "name": "yolopafpn",
                "params": {
                    "dep_mul": 1.33,
                    "act_type": "silu",
                },
            },
            head={
                "name": "anchor_free_decoupled_head",
                "params": {
                    "act_type": "silu",
                    # postprocessor - decode
                    "score_thresh": 0.01,
                    # postprocessor - nms
                    "nms_thresh": 0.65,
                    "class_agnostic": False,
                },
            },
        )
    )
    losses: List[Dict[str, Any]] = field(
        default_factory=lambda: [{"criterion": "yolox_loss", "weight": None, "l1_activate_epoch": 1}]
    )


@dataclass
class DetectionYoloXLModelConfig(ModelConfig):
    task: str = "detection"
    name: str = "yolox_l"
    architecture: ArchitectureConfig = field(
        default_factory=lambda: CSPDarkNetLArchitectureConfig(
            neck={
                "name": "yolopafpn",
                "params": {
                    "dep_mul": 1.0,
                    "act_type": "silu",
                },
            },
            head={
                "name": "anchor_free_decoupled_head",
                "params": {
                    "act_type": "silu",
                    # postprocessor - decode
                    "score_thresh": 0.01,
                    # postprocessor - nms
                    "nms_thresh": 0.65,
                    "class_agnostic": False,
                },
            },
        )
    )
    losses: List[Dict[str, Any]] = field(
        default_factory=lambda: [{"criterion": "yolox_loss", "weight": None, "l1_activate_epoch": 1}]
    )


@dataclass
class DetectionYoloXMModelConfig(ModelConfig):
    task: str = "detection"
    name: str = "yolox_m"
    architecture: ArchitectureConfig = field(
        default_factory=lambda: CSPDarkNetMArchitectureConfig(
            neck={
                "name": "yolopafpn",
                "params": {
                    "dep_mul": 0.67,
                    "act_type": "silu",
                },
            },
            head={
                "name": "anchor_free_decoupled_head",
                "params": {
                    "act_type": "silu",
                    # postprocessor - decode
                    "score_thresh": 0.01,
                    # postprocessor - nms
                    "nms_thresh": 0.65,
                    "class_agnostic": False,
                },
            },
        )
    )
    losses: List[Dict[str, Any]] = field(
        default_factory=lambda: [{"criterion": "yolox_loss", "weight": None, "l1_activate_epoch": 1}]
    )


@dataclass
class DetectionYoloXSModelConfig(ModelConfig):
    task: str = "detection"
    name: str = "yolox_s"
    architecture: ArchitectureConfig = field(
        default_factory=lambda: CSPDarkNetSArchitectureConfig(
            neck={
                "name": "yolopafpn",
                "params": {
                    "dep_mul": 0.33,
                    "act_type": "silu",
                },
            },
            head={
                "name": "anchor_free_decoupled_head",
                "params": {
                    "act_type": "silu",
                    # postprocessor - decode
                    "score_thresh": 0.01,
                    # postprocessor - nms
                    "nms_thresh": 0.65,
                    "class_agnostic": False,
                },
            },
        )
    )
    losses: List[Dict[str, Any]] = field(
        default_factory=lambda: [{"criterion": "yolox_loss", "weight": None, "l1_activate_epoch": 1}]
    )
