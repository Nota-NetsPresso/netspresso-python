from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class AugmentationSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    parameters: dict
    phase: str
    hyperparameter_id: int


class DatasetSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    format: str
    root_path: str
    train_path: str
    valid_path: Optional[str]
    test_path: Optional[str]
    storage_location: str
    train_valid_split_ratio: float
    id_mapping: Optional[List]
    palette: Optional[dict]


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


class EnvironmentSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    seed: int
    num_workers: int
    gpus: str


class PerformanceSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    train_losses: dict
    valid_losses: dict
    train_metrics: dict
    valid_metrics: dict
    metrics_list: List[str]
    primary_metric: str
    flops: int
    params: int
    total_train_time: float
    best_epoch: int
    last_epoch: int
    total_epoch: int
    status: str


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
