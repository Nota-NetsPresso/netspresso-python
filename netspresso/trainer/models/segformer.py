from dataclasses import dataclass, field
from typing import Any, Dict, List

from netspresso.trainer.models.base import ArchitectureConfig, ModelConfig


@dataclass
class SegFormerB0ArchitectureConfig(ArchitectureConfig):
    backbone: Dict[str, Any] = field(
        default_factory=lambda: {
            "name": "mixtransformer",
            "params": {
                "ffn_intermediate_expansion_ratio": 4,
                "ffn_act_type": "gelu",
                "ffn_dropout_prob": 0.0,
                "attention_dropout_prob": 0.0,
            },
            "stage_params": [
                {
                    "num_blocks": 2,
                    "sequence_reduction_ratio": 8,
                    "attention_chananels": 32,
                    "embedding_patch_sizes": 7,
                    "embedding_strides": 4,
                    "num_attention_heads": 1,
                },
                {
                    "num_blocks": 2,
                    "sequence_reduction_ratio": 4,
                    "attention_chananels": 64,
                    "embedding_patch_sizes": 3,
                    "num_attention_heads": 2,
                },
                {
                    "num_blocks": 2,
                    "sequence_reduction_ratio": 2,
                    "attention_chananels": 160,
                    "embedding_patch_sizes": 3,
                    "embedding_strides": 2,
                    "num_attention_heads": 5,
                },
                {
                    "num_blocks": 2,
                    "sequence_reduction_ratio": 1,
                    "attention_chananels": 256,
                    "embedding_patch_sizes": 3,
                    "embedding_strides": 2,
                    "num_attention_heads": 8,
                },
            ],
        }
    )


@dataclass
class SegmentationSegFormerB0ModelConfig(ModelConfig):
    task: str = "segmentation"
    name: str = "segformer_b0"
    architecture: ArchitectureConfig = field(
        default_factory=lambda: SegFormerB0ArchitectureConfig(
            head={
                "name": "all_mlp_decoder",
                "params": {
                    "intermediate_channels": 256,
                    "classifier_dropout_prob": 0.0,
                    "resize_output": [512, 512],
                },
            }
        )
    )
    losses: List[Dict[str, Any]] = field(
        default_factory=lambda: [{"criterion": "seg_cross_entropy", "ignore_index": 255, "weight": None}]
    )
