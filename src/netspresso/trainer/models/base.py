from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from omegaconf import MISSING


@dataclass
class ArchitectureConfig:
    full: Optional[Dict[str, Any]] = None
    backbone: Optional[Dict[str, Any]] = None
    neck: Optional[Dict[str, Any]] = None
    head: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        assert bool(self.full) != bool(self.backbone), "Only one of full or backbone should be given."


@dataclass
class CheckpointConfig:
    use_pretrained: bool = True
    load_head: bool = False
    path: Optional[Union[Path, str]] = None
    fx_model_path: Optional[Union[Path, str]] = None
    optimizer_path: Optional[Union[Path, str]] = None


@dataclass
class ModelConfig:
    task: str = MISSING
    name: str = MISSING
    checkpoint: CheckpointConfig = field(default_factory=lambda: CheckpointConfig())
    freeze_backbone: bool = False
    architecture: ArchitectureConfig = field(default_factory=lambda: ArchitectureConfig())
    postprocessor: Optional[Dict[str, Any]] = None
    losses: Optional[List[Dict[str, Any]]] = None
