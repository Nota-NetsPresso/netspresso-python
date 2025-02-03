from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_serializer

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
    display_name: Optional[PretrainedModelDisplay] = Field(description="Pretrained model display name")
    group_name: Optional[PretrainedModelGroup] = Field(description="Pretrained model group name")

    @field_serializer('display_name')
    def serialize_display_name(self, value: Optional[PretrainedModelDisplay]) -> PretrainedModelDisplay:
        return MODEL_DISPLAY_MAP.get(self.name)

    @field_serializer('group_name')
    def serialize_group_name(self, value: Optional[PretrainedModelGroup]) -> PretrainedModelGroup:
        return MODEL_GROUP_MAP.get(self.name)


class TaskPayload(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: Task = Field(description="Task name")
    display_name: Optional[TaskDisplay] = Field(description="Task display name")

    @field_serializer('display_name')
    def serialize_display_name(self, value: Optional[TaskDisplay]) -> TaskDisplay:
        return TASK_DISPLAY_MAP.get(self.name)


class FrameworkPayload(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: Framework = Field(description="Framework name")
    display_name: Optional[FrameworkDisplay] = Field(description="Framework display name")

    @field_serializer('display_name')
    def serialize_display_name(self, value: Optional[FrameworkDisplay]) -> FrameworkDisplay:
        return FRAMEWORK_DISPLAY_MAP.get(self.name)


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
