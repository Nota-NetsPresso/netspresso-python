from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from app.api.v1.schemas.base import ResponseItem

from .dataset import DatasetCreate, DatasetSchema
from .environment import EnvironmentCreate, EnvironmentSchema
from .hyperparameter import HyperparameterCreate, HyperparameterSchema
from .performance import PerformanceSchema


class TrainTaskCreate(BaseModel):
    project_id: str
    pretrained_model_name: str
    task: str
    framework: str
    input_shapes: List[Dict]
    img_size: int
    dataset: Optional[DatasetCreate]
    hyperparameter: Optional[HyperparameterCreate]
    environment: Optional[EnvironmentCreate]


class TrainTaskSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    task_id: str
    pretrained_model_name: str
    task: str
    framework: str
    input_shapes: List[Dict]
    status: str
    error_detail: Optional[Dict] = None
    dataset: Optional[DatasetSchema]
    hyperparameter: Optional[HyperparameterSchema]
    environment: Optional[EnvironmentSchema]
    performance: Optional[PerformanceSchema]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class TrainTaskDetailResponse(ResponseItem):
    data: TrainTaskSchema
