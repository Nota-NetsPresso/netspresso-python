from dataclasses import dataclass, field
from typing import Any, Dict, List

from netspresso.trainer.models.base import ArchitectureConfig, CheckpointConfig, ModelConfig


@dataclass
class MobileViTArchitectureConfig(ArchitectureConfig):
    backbone: Dict[str, Any] = field(
        default_factory=lambda: {
            "name": "mobilevit",
            "params": {
                "patch_size": 2,
                "num_attention_heads": 4,
                "attention_dropout_prob": 0.1,
                "ffn_dropout_prob": 0.0,
                "output_expansion_ratio": 4,
                "use_fusion_layer": True,
            },
            "stage_params": [
                {
                    "out_channels": 32,
                    "block_type": "mv2",
                    "num_blocks": 1,
                    "stride": 1,
                    "ir_expansion_ratio": 4,  # [mv2_exp_mult] * 4
                },
                {
                    "block_type": "mv2",
                    "out_channels": 64,
                    "num_blocks": 3,
                    "stride": 2,
                    "ir_expansion_ratio": 4,  # [mv2_exp_mult] * 4
                },
                {
                    "block_type": "mobilevit",
                    "out_channels": 96,
                    "num_blocks": 2,
                    "stride": 2,
                    "hidden_size": 144,
                    "intermediate_size": 288,
                    "dilate": False,
                    "ir_expansion_ratio": 4,  # [mv2_exp_mult] * 4
                },
                {
                    "block_type": "mobilevit",
                    "out_channels": 128,
                    "num_blocks": 4,
                    "stride": 2,
                    "hidden_size": 192,
                    "intermediate_size": 384,
                    "dilate": False,
                    "ir_expansion_ratio": 4,  # [mv2_exp_mult] * 4
                },
                {
                    "block_type": "mobilevit",
                    "out_channels": 160,
                    "num_blocks": 3,
                    "stride": 2,
                    "hidden_size": 240,
                    "intermediate_size": 480,
                    "dilate": False,
                    "ir_expansion_ratio": 4,  # [mv2_exp_mult] * 4
                },
            ],
        }
    )


@dataclass
class ClassificationMobileViTModelConfig(ModelConfig):
    task: str = "classification"
    name: str = "mobilevit_s"
    architecture: ArchitectureConfig = field(
        default_factory=lambda: MobileViTArchitectureConfig(
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
