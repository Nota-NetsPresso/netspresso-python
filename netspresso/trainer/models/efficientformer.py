from dataclasses import dataclass, field
from typing import Any, Dict, List

from netspresso.trainer.models.base import ArchitectureConfig, CheckpointConfig, ModelConfig


@dataclass
class EfficientFormerArchitectureConfig(ArchitectureConfig):
    backbone: Dict[str, Any] = field(
        default_factory=lambda: {
            "name": "efficientformer",
            "params": {
                "num_attention_heads": 8,
                "attention_channels": 256,
                "attention_dropout_prob": 0.0,
                "attention_value_expansion_ratio": 4,
                "ffn_intermediate_ratio": 4,
                "ffn_dropout_prob": 0.0,
                "ffn_act_type": "gelu",
                "vit_num": 1,
            },
            "stage_params": [
                {"num_blocks": 3, "channels": 48},
                {"num_blocks": 2, "channels": 96},
                {"num_blocks": 6, "channels": 224},
                {"num_blocks": 4, "channels": 448},
            ],
        }
    )


@dataclass
class ClassificationEfficientFormerModelConfig(ModelConfig):
    task: str = "classification"
    name: str = "efficientformer_l1"
    architecture: ArchitectureConfig = field(
        default_factory=lambda: EfficientFormerArchitectureConfig(
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
    losses: List[Dict[str, Any]] = field(
        default_factory=lambda: [{"criterion": "cross_entropy", "label_smoothing": 0.1, "weight": None}]
    )


@dataclass
class SegmentationEfficientFormerModelConfig(ModelConfig):
    task: str = "segmentation"
    name: str = "efficientformer_l1"
    architecture: ArchitectureConfig = field(
        default_factory=lambda: EfficientFormerArchitectureConfig(
            head={
                "name": "all_mlp_decoder",
                "params": {
                    "intermediate_channels": 256,
                    "classifier_dropout_prob": 0.0,
                },
            }
        )
    )
    losses: List[Dict[str, Any]] = field(
        default_factory=lambda: [{"criterion": "seg_cross_entropy", "ignore_index": 255, "weight": None}]
    )


@dataclass
class DetectionEfficientFormerModelConfig(ModelConfig):
    task: str = "detection"
    name: str = "efficientformer_l1"
    architecture: ArchitectureConfig = field(
        default_factory=lambda: EfficientFormerArchitectureConfig(
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
                    # postprocessor - decode
                    "topk_candidates": 1000,
                    "score_thresh": 0.05,
                    # postprocessor - nms
                    "nms_thresh": 0.45,
                    "class_agnostic": False,
                },
            },
        )
    )
    losses: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {"criterion": "retinanet_loss", "weight": None},
        ]
    )
