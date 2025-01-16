from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from netspresso.enums.train import Optimizer, Scheduler

from .augmentation import AugmentationPayload


class SupportedModel(BaseModel):
    classification: List[str] = Field(..., description="Supported models for classification tasks")
    detection: List[str] = Field(..., description="Supported models for object detection tasks")
    segmentation: List[str] = Field(..., description="Supported models for semantic segmentation tasks")


class SupportedModelResponse(BaseModel):
    data: SupportedModel = Field(..., description="Supported models for classification tasks")


class OptimizerPayload(BaseModel):
    name: Optimizer = Field(Optimizer.ADAM, description="Name of the optimizer")
    display_name: Optional[str] = Field(Optimizer.to_display_name(Optimizer.ADAM), description="Display name of the optimizer")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters and their default values for the optimizer")

    @model_validator(mode="after")
    def set_display_name(cls, values):
        if not values.display_name:
            values.display_name = Optimizer.to_display_name(values.name)
        return values


class SupportedOptimizersResponse(BaseModel):
    data: List[OptimizerPayload] = Field(..., description="Supported optimizers for training tasks")


class SchedulerPayload(BaseModel):
    name: Scheduler = Field(Scheduler.COSINE_ANNEALING_WARM_RESTARTS_WITH_CUSTOM_WARM_UP, description="Name of the scheduler")
    display_name: Optional[str] = Field(Scheduler.to_display_name(Scheduler.COSINE_ANNEALING_WARM_RESTARTS_WITH_CUSTOM_WARM_UP), description="Display name of the scheduler")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters and their default values for the scheduler")

    @model_validator(mode="after")
    def set_display_name(cls, values):
        if not values.display_name:
            values.display_name = Scheduler.to_display_name(values.name)
        return values


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
