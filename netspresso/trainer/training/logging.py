from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union


@dataclass
class LoggingConfig:
    project_id: Optional[str] = None
    output_dir: Union[Path, str] = "./outputs"
    tensorboard: bool = True
    image: bool = True
    stdout: bool = True
    save_optimizer_state: bool = True
    onnx_input_size: List = field(default_factory=lambda: [512, 512])
    validation_epoch: int = 10
    save_checkpoint_epoch: Optional[int] = None

    def __post_init__(self):
        if self.save_checkpoint_epoch is None:
            self.save_checkpoint_epoch = self.validation_epoch
