from dataclasses import dataclass, field
from typing import Dict, List

from netspresso.enums.metadata import TaskType
from netspresso.metadata.common import AvailableOption, BaseMetadata, InputShape


@dataclass
class ModelInfo:
    task: str = ""
    model: str = ""
    dataset: str = ""
    input_shapes: List[InputShape] = field(default_factory=lambda: [InputShape()])


@dataclass
class TrainingInfo:
    epochs: int = 0
    batch_size: int = 0
    learning_rate: float = 0.001
    optimizer: str = "adam"


@dataclass
class TrainerMetadata(BaseMetadata):
    task_type: TaskType = TaskType.TRAIN
    logging_dir: str = ""
    best_fx_model_path: str = ""
    best_onnx_model_path: str = ""
    hparams: str = ""
    model_info: ModelInfo = field(default_factory=ModelInfo)
    training_info: TrainingInfo = field(default_factory=TrainingInfo)
    traning_result: Dict = field(default_factory=dict)
    available_options: List[AvailableOption] = field(default_factory=list)

    def update_model_info(self, task, model, dataset, input_shapes):
        self.model_info.task = task
        self.model_info.model = model
        self.model_info.dataset = dataset
        self.model_info.input_shapes = input_shapes

    def update_training_info(self, epochs, batch_size, learning_rate, optimizer):
        self.training_info.epochs = epochs
        self.training_info.batch_size = batch_size
        self.training_info.learning_rate = learning_rate
        self.training_info.optimizer = optimizer

    def update_training_result(self, training_summary):
        self.traning_result = training_summary

    def update_logging_dir(self, logging_dir):
        self.logging_dir = logging_dir

    def update_best_fx_model_path(self, best_fx_model_path):
        self.best_fx_model_path = best_fx_model_path

    def update_best_onnx_model_path(self, best_onnx_model_path):
        self.best_onnx_model_path = best_onnx_model_path

    def update_hparams(self, hparams):
        self.hparams = hparams

    def update_available_options(self, available_options):
        self.available_options = [available_option.to() for available_option in available_options]