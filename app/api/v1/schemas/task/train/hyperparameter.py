from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from netspresso.enums.train import Optimizer, Scheduler

from .augmentation import AugmentationPayload


class TrainerModel(BaseModel):
    name: str = Field(..., description="Name of the model")
    display_name: Optional[str] = Field(..., description="Display name of the model")


class SupportedModel(BaseModel):
    classification: List[TrainerModel] = Field(..., description="Supported models for classification tasks")
    detection: List[TrainerModel] = Field(..., description="Supported models for object detection tasks")
    segmentation: List[TrainerModel] = Field(..., description="Supported models for semantic segmentation tasks")


class SupportedModelResponse(BaseModel):
    data: SupportedModel = Field(..., description="Supported models for classification tasks")


class OptimizerPayload(BaseModel):
    name: Optimizer = Field(Optimizer.ADAM, description="Name of the optimizer")
    display_name: Optional[str] = Field(Optimizer.to_display_name(Optimizer.ADAM), description="Display name of the optimizer")


class SupportedOptimizersResponse(BaseModel):
    data: List[OptimizerPayload] = Field(..., description="Supported optimizers for training tasks")


class SchedulerPayload(BaseModel):
    name: Scheduler = Field(Scheduler.COSINE_ANNEALING_WARM_RESTARTS_WITH_CUSTOM_WARM_UP, description="Name of the scheduler")
    display_name: Optional[str] = Field(Scheduler.to_display_name(Scheduler.COSINE_ANNEALING_WARM_RESTARTS_WITH_CUSTOM_WARM_UP), description="Display name of the scheduler")


class SupportedSchedulersResponse(BaseModel):
    data: List[SchedulerPayload] = Field(..., description="Supported schedulers for training tasks")


class HyperparameterCreate(BaseModel):
    epochs: int = Field(default=10, description="Number of epochs to train for")
    batch_size: int = Field(default=32, description="Batch size to use")
    learning_rate: float = Field(default=0.001, description="Learning rate to use")
    optimizer: OptimizerPayload = Field(default_factory=OptimizerPayload, description="Optimizer to use")
    scheduler: SchedulerPayload = Field(default_factory=SchedulerPayload, description="Scheduler to use")
    augmentations: Optional[List[AugmentationPayload]] = Field(default=None, description="List of augmentations to apply")


class HyperparameterPayload(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: OptimizerPayload
    scheduler: SchedulerPayload
    augmentations: List[AugmentationPayload] = []
