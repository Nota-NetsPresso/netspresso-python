from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.api.v1.schemas.base import ResponseItem

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
    pretrained_model_name: str
    task: str
    framework: str
    input_shapes: List[InputShape] = Field(default_factory=list, description="List of input shapes")
    dataset: Optional[DatasetCreate]
    hyperparameter: Optional[HyperparameterCreate]
    environment: Optional[EnvironmentCreate]


class TrainTaskSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    task_id: str
    model_id: Optional[str] = None
    pretrained_model_name: str
    task: str
    framework: str
    input_shapes: List[Dict]
    status: str
    error_detail: Optional[Dict] = None
    dataset: Optional[DatasetPayload]
    hyperparameter: Optional[HyperparameterPayload]
    environment: Optional[EnvironmentPayload]
    performance: Optional[PerformancePayload]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class TrainTaskDetailResponse(ResponseItem):
    data: TrainTaskSchema
