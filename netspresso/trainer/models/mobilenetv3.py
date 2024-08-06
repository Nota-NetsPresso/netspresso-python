from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from netspresso.trainer.models.base import ArchitectureConfig, CheckpointConfig, ModelConfig


@dataclass
class MobileNetV3SmallArchitectureConfig(ArchitectureConfig):
    backbone: Dict[str, Any] = field(
        default_factory=lambda: {
            "name": "mobilenetv3",
            "params": None,
            "stage_params": [
                {
                    "in_channels": [16],
                    "kernel_sizes": [3],
                    "expanded_channels": [16],
                    "out_channels": [16],
                    "use_se": [True],
                    "act_type": ["relu"],
                    "stride": [2],
                },
                {
                    "in_channels": [16, 24],
                    "kernel_sizes": [3, 3],
                    "expanded_channels": [72, 88],
                    "out_channels": [24, 24],
                    "use_se": [False, False],
                    "act_type": ["relu", "relu"],
                    "stride": [2, 1],
                },
                {
                    "in_channels": [24, 40, 40, 40, 48],
                    "kernel_sizes": [5, 5, 5, 5, 5],
                    "expanded_channels": [96, 240, 240, 120, 144],
                    "out_channels": [40, 40, 40, 48, 48],
                    "use_se": [True, True, True, True, True],
                    "act_type": ["hard_swish", "hard_swish", "hard_swish", "hard_swish", "hard_swish"],
                    "stride": [2, 1, 1, 1, 1],
                },
                {
                    "in_channels": [48, 96, 96],
                    "kernel_sizes": [5, 5, 5],
                    "expanded_channels": [288, 576, 576],
                    "out_channels": [96, 96, 96],
                    "use_se": [True, True, True],
                    "act_type": ["hard_swish", "hard_swish", "hard_swish"],
                    "stride": [2, 1, 1],
                },
            ],
        }
    )


@dataclass
class MobileNetV3LargeArchitectureConfig(ArchitectureConfig):
    backbone: Dict[str, Any] = field(
        default_factory=lambda: {
            "name": "mobilenetv3",
            "params": None,
            "stage_params": [
                {
                    "in_channels": [16, 16, 24],
                    "kernel_sizes": [3, 3, 3],
                    "expanded_channels": [16, 64, 72],
                    "out_channels": [16, 24, 24],
                    "use_se": [False, False, False],
                    "act_type": ["relu", "relu", "relu"],
                    "stride": [1, 2, 1],
                },
                {
                    "in_channels": [24, 40, 40],
                    "kernel_sizes": [5, 5, 5],
                    "expanded_channels": [72, 120, 120],
                    "out_channels": [40, 40, 40],
                    "use_se": [True, True, True],
                    "act_type": ["relu", "relu", "relu"],
                    "stride": [2, 1, 1],
                },
                {
                    "in_channels": [40, 80, 80, 80, 80, 112],
                    "kernel_sizes": [3, 3, 3, 3, 3, 3],
                    "expanded_channels": [240, 200, 184, 184, 480, 672],
                    "out_channels": [80, 80, 80, 80, 112, 112],
                    "use_se": [False, False, False, False, True, True],
                    "act_type": ["hard_swish", "hard_swish", "hard_swish", "hard_swish", "hard_swish", "hard_swish"],
                    "stride": [2, 1, 1, 1, 1, 1],
                },
                {
                    "in_channels": [112, 160, 160],
                    "kernel_sizes": [5, 5, 5],
                    "expanded_channels": [672, 960, 960],
                    "out_channels": [160, 160, 160],
                    "use_se": [True, True, True],
                    "act_type": ["hard_swish", "hard_swish", "hard_swish"],
                    "stride": [2, 1, 1],
                },
            ],
        }
    )


@dataclass
class ClassificationMobileNetV3LargeModelConfig(ModelConfig):
    task: str = "classification"
    name: str = "mobilenet_v3_large"
    architecture: ArchitectureConfig = field(
        default_factory=lambda: MobileNetV3LargeArchitectureConfig(
            head={
                "name": "fc",
                "params": {
                    "num_layers": 2,
                    "intermediate_channels": 1200,
                    "act_type": "hard_swish",
                    "dropout_prob": 0.0,
                },
            }
        )
    )
    postprocessor: Optional[Dict[str, Any]] = None
    losses: List[Dict[str, Any]] = field(
        default_factory=lambda: [{"criterion": "cross_entropy", "label_smoothing": 0.1, "weight": None}]
    )


@dataclass
class ClassificationMobileNetV3SmallModelConfig(ModelConfig):
    task: str = "classification"
    name: str = "mobilenet_v3_small"
    architecture: ArchitectureConfig = field(
        default_factory=lambda: MobileNetV3SmallArchitectureConfig(
            head={
                "name": "fc",
                "params": {
                    "num_layers": 1,
                    "intermediate_channels": None,
                    "act_type": None,
                    "dropout_prob": 0.0,
                },
            }
        )
    )
    postprocessor: Optional[Dict[str, Any]] = None
    losses: List[Dict[str, Any]] = field(
        default_factory=lambda: [{"criterion": "cross_entropy", "label_smoothing": 0.1, "weight": None}]
    )


@dataclass
class SegmentationMobileNetV3SmallModelConfig(ModelConfig):
    task: str = "segmentation"
    name: str = "mobilenet_v3_small"
    architecture: ArchitectureConfig = field(
        default_factory=lambda: MobileNetV3SmallArchitectureConfig(
            head={
                "name": "all_mlp_decoder",
                "params": {
                    "intermediate_channels": 256,
                    "classifier_dropout_prob": 0.0,
                },
            }
        )
    )
    postprocessor: Optional[Dict[str, Any]] = None
    losses: List[Dict[str, Any]] = field(
        default_factory=lambda: [{"criterion": "seg_cross_entropy", "ignore_index": 255, "weight": None}]
    )


@dataclass
class DetectionMobileNetV3SmallModelConfig(ModelConfig):
    task: str = "detection"
    name: str = "mobilenet_v3_small"
    architecture: ArchitectureConfig = field(
        default_factory=lambda: MobileNetV3SmallArchitectureConfig(
            neck={
                "name": "fpn",
                "params": {
                    "num_outs": 4,
                    "start_level": 0,
                    "end_level": -1,
                    "add_extra_convs": False,
                    "relu_before_extra_convs": False,
                },
            },
            head={
                "name": "anchor_decoupled_head",
                "params": {
                    # Anchor parameters
                    "anchor_sizes": [
                        [
                            32,
                        ],
                        [
                            64,
                        ],
                        [
                            128,
                        ],
                        [
                            256,
                        ],
                    ],
                    "aspect_ratios": [0.5, 1.0, 2.0],
                    "num_layers": 1,
                    "norm_type": "batch_norm",
                },
            },
        )
    )
    postprocessor: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {
            "params": {
                # postprocessor - decode
                "topk_candidates": 1000,
                "score_thresh": 0.05,
                # postprocessor - nms
                "nms_thresh": 0.45,
                "class_agnostic": False,
            },
        }
    )
    losses: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {"criterion": "retinanet_loss", "weight": None},
        ]
    )
