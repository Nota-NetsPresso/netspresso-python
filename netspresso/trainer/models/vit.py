from dataclasses import dataclass, field
from typing import Any, Dict, List

from netspresso.trainer.models.base import ArchitectureConfig, ModelConfig


@dataclass
class ViTTinyArchitectureConfig(ArchitectureConfig):
    backbone: Dict[str, Any] = field(
        default_factory=lambda: {
            "name": "vit",
            "params": {
                "patch_size": 16,
                "attention_channels": 192,
                "num_blocks": 12,
                "num_attention_heads": 3,
                "attention_dropout_prob": 0.0,
                "ffn_intermediate_channels": 768,
                "ffn_dropout_prob": 0.1,
                "use_cls_token": True,
                "vocab_size": 1000,
            },
            "stage_params": None,
        }
    )


@dataclass
class ClassificationViTTinyModelConfig(ModelConfig):
    task: str = "classification"
    name: str = "vit_tiny"
    architecture: ArchitectureConfig = field(
        default_factory=lambda: ViTTinyArchitectureConfig(
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
