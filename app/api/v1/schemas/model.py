from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.api.v1.schemas.base import ResponseItem, ResponsePaginationItems
from netspresso.enums import Status


class ExperimentStatus(BaseModel):
    convert: Status = Field(default=Status.NOT_STARTED, description="The status of the conversion experiment.")
    benchmark: Status = Field(default=Status.NOT_STARTED, description="The status of the benchmark experiment.")


class ModelPayload(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    model_id: str = Field(..., description="The unique identifier for the model.")
    name: str = Field(..., description="The name of the model.")
    type: str = Field(..., description="The type of the model (e.g., trained_model, compressed_model).")
    is_retrainable: bool
    status: Status = Field(default=Status.NOT_STARTED, description="The current status of the model.")
    train_task_id: str
    project_id: str
    user_id: str
    compress_task_ids: Optional[List] = []
    convert_task_ids: Optional[List] = []
    benchmark_task_ids: Optional[List] = []
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    latest_experiments: ExperimentStatus = Field(default_factory=ExperimentStatus)


class ExperimentStatusResponse(ResponseItem):
    data: ExperimentStatus


class ModelDetailResponse(ResponseItem):
    data: ModelPayload


class ModelsResponse(ResponsePaginationItems):
    data: List[ModelPayload]
