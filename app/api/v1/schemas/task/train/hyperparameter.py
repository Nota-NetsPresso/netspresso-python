from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from netspresso.enums.train import Optimizer, Scheduler

from .augmentation import AugmentationSchema


class OptimizerResponse(BaseModel):
    name: str = Field(..., description="Name of the optimizer")
    parameters: Dict[str, Any] = Field(..., description="Parameters and their default values for the optimizer")


class SchedulerResponse(BaseModel):
    name: str = Field(..., description="Name of the scheduler")
    parameters: Dict[str, Any] = Field(..., description="Parameters and their default values for the scheduler")


class HyperparameterCreate(BaseModel):
    epochs: int = Field(default=10, description="Number of epochs to train for")
    batch_size: int = Field(default=32, description="Batch size to use")
    learning_rate: float = Field(default=0.001, description="Learning rate to use")
    optimizer_name: Optimizer = Field(default=Optimizer.ADAM, description="Optimizer to use")
    optimizer_params: Optional[dict] = Field(default_factory=dict, description="Parameters for the optimizer")
    scheduler_name: Scheduler = Field(default=Scheduler.COSINE_ANNEALING_WARM_RESTARTS_WITH_CUSTOM_WARM_UP, description="Scheduler to use")
    scheduler_params: Optional[dict] = Field(default_factory=dict, description="Parameters for the scheduler")
    augmentations: List[AugmentationSchema] = Field(default_factory=list, description="List of augmentations to apply")


class HyperparameterSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    epochs: int
    batch_size: int
    learning_rate: float
    optimizer_name: str
    optimizer_params: Optional[dict]
    scheduler_name: str
    scheduler_params: Optional[dict]
    augmentations: List[AugmentationSchema] = []
