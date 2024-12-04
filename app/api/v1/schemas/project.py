from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.api.v1.schemas.base import ResponseItem, ResponsePaginationItems
from netspresso.enums import Status


class ProjectCreate(BaseModel):
    project_name: str = Field(..., description="The name of the project to be created.")

    @field_validator("project_name")
    def validate_length_of_project_name(cls, project_name: str) -> str:
        if len(project_name) > 30:
            raise ValueError("The project_name can't exceed 30 characters.")
        return project_name


class ProjectSummaryPayload(ProjectCreate):
    model_config = ConfigDict(from_attributes=True)

    project_id: str = Field(..., description="The unique identifier for the project.")
    user_id: str = Field(..., description="The unique identifier for the user associated with the project.")


class ExperimentStatus(BaseModel):
    convert: Status = Field(default=Status.NOT_STARTED, description="The status of the conversion experiment.")
    benchmark: Status = Field(default=Status.NOT_STARTED, description="The status of the benchmark experiment.")


class ModelSummary(BaseModel):
    model_id: str = Field(..., description="The unique identifier for the model.")
    name: str = Field(..., description="The name of the model.")
    type: str = Field(..., description="The type of the model (e.g., trained_model, compressed_model).")
    status: Status = Field(default=Status.NOT_STARTED, description="The current status of the model.")
    latest_experiments: ExperimentStatus = Field(..., description="The latest status of experiments for the model.")


class ProjectDetailPayload(ProjectSummaryPayload):
    models: List[ModelSummary] = Field(..., description="The list of models associated with the project.")


class ProjectResponse(ResponseItem):
    data: ProjectSummaryPayload


class ProjectDetailResponse(ResponseItem):
    data: ProjectDetailPayload


class ProjectsResponse(ResponsePaginationItems):
    data: List[ProjectSummaryPayload]
