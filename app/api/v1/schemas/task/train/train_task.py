from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.api.v1.schemas.base import ResponseItem
from netspresso.enums.train import (
    Framework, 
    FrameworkDisplay, 
    Task, 
    TaskDisplay,
    PretrainedModel,
    PretrainedModelDisplay,
    PretrainedModelGroup,
    TASK_DISPLAY_MAP,
    FRAMEWORK_DISPLAY_MAP,
    MODEL_DISPLAY_MAP,
    MODEL_GROUP_MAP,
    TASK_DISPLAY_MAP,
    FRAMEWORK_DISPLAY_MAP,
)

from .dataset import DatasetCreate, DatasetPayload
from .environment import EnvironmentCreate, EnvironmentPayload
from .hyperparameter import HyperparameterCreate, HyperparameterPayload
from .performance import PerformancePayload


class InputShape(BaseModel):
    batch: int = Field(default=1, description="Batch size")
    channel: int = Field(default=3, description="Number of channels")
    dimension: List[int] = Field(default=[224, 224], description="Input shape")


class TrainingCreate(BaseModel):
    project_id: str
    name: str
    pretrained_model: str
    task: Task = Field(default=Task.OBJECT_DETECTION, description="Task")
    input_shapes: List[InputShape] = Field(default_factory=list, description="List of input shapes")
    dataset: Optional[DatasetCreate]
    hyperparameter: Optional[HyperparameterCreate]
    environment: Optional[EnvironmentCreate]


class PretrainedModelPayload(BaseModel):
    name: PretrainedModel = Field(description="Pretrained model name")
    display_name: PretrainedModelDisplay = Field(description="Pretrained model display name")
    group_name: PretrainedModelGroup = Field(description="Pretrained model group name")

    @model_validator(mode='before')
    def set_display_and_group_name(cls, values) -> str:
        values.display_name = MODEL_DISPLAY_MAP.get(values.name)
        values.group_name = MODEL_GROUP_MAP.get(values.name)
        return values


class TaskPayload(BaseModel):
    name: Task = Field(description="Task name")
    display_name: TaskDisplay = Field(description="Task display name")

    @model_validator(mode='before')
    def set_display_name(cls, values) -> str:
        values.display_name = TASK_DISPLAY_MAP.get(values.name)
        return values


class FrameworkPayload(BaseModel):
    name: Framework = Field(description="Framework name")
    display_name: FrameworkDisplay = Field(description="Framework display name")

    @model_validator(mode='before')
    def set_display_name(cls, values) -> str:
        values.display_name = FRAMEWORK_DISPLAY_MAP.get(values.name)
        return values


class TrainingPayload(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    task_id: str
    model_id: Optional[str] = None
    pretrained_model: PretrainedModelPayload
    task: TaskPayload
    framework: FrameworkPayload
    input_shapes: List[Dict]
    status: str
    error_detail: Optional[Dict] = None
    dataset: Optional[DatasetPayload]
    hyperparameter: Optional[HyperparameterPayload]
    performance: Optional[PerformancePayload]
    environment: Optional[EnvironmentPayload]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class TrainingResponse(ResponseItem):
    data: TrainingPayload
