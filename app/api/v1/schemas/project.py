from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.api.v1.schemas.base import ResponseItem, ResponsePaginationItems


class ProjectCreate(BaseModel):
    project_name: str = Field(..., description="The name of the project to be created.")

    @field_validator("project_name")
    def validate_length_of_project_name(cls, project_name: str) -> str:
        if len(project_name) > 30:
            raise ValueError("The project_name can't exceed 30 characters.")
        return project_name


class ProjectPayload(ProjectCreate):
    model_config = ConfigDict(from_attributes=True)

    project_id: str = Field(..., description="The unique identifier for the project.")
    user_id: str = Field(..., description="The unique identifier for the user associated with the project.")


class ProjectResponse(ResponseItem):
    data: ProjectPayload


class ProjectsResponse(ResponsePaginationItems):
    data: List[ProjectPayload]
