from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.api.v1.schemas.base import ResponseItem, ResponsePaginationItems
from app.api.v1.schemas.train_task import TrainTaskSchema
from netspresso.enums import Status


class ModelDetailPayload(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    model_id: str = Field(..., description="The unique identifier for the model.")
    name: str = Field(..., description="The name of the model.")
    type: str = Field(..., description="The type of the model (e.g., trained_model, compressed_model).")
    is_retrainable: bool
    status: Status = Field(default=Status.NOT_STARTED, description="The current status of the model.")
    train_task_id: str
    train_task: TrainTaskSchema
    project_id: str
    user_id: str
    compress_tasks: Optional[List] = []
    convert_tasks: Optional[List] = []
    benchmark_tasks: Optional[List] = []
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ModelDetailResponse(ResponseItem):
    data: ModelDetailPayload


class ModelsResponse(ResponsePaginationItems):
    data: List[ModelDetailPayload]
