from typing import Optional
from uuid import uuid4

from fastapi import APIRouter

from app.api.v1.schemas.project import (
    ExperimentStatus,
    ModelSummary,
    ProjectCreate,
    ProjectDetailPayload,
    ProjectDetailResponse,
    ProjectPayload,
    ProjectResponse,
    ProjectsResponse,
)

router = APIRouter()


@router.post("", response_model=ProjectResponse)
def create_project(
    *,
    request_body: ProjectCreate,
) -> ProjectResponse:

    project = ProjectPayload(
        project_id=str(uuid4()),
        project_name=request_body.project_name,
        user_id=str(uuid4()),
    )

    return ProjectResponse(data=project)


@router.get("", response_model=ProjectsResponse)
def get_projects(
    *,
    user_id: str,
    skip: Optional[int] = 0,
    limit: Optional[int] = 100,
) -> ProjectsResponse:

    projects = [
        ProjectPayload(
            project_id=str(uuid4()),
            project_name="project_test_1",
            user_id=str(uuid4()),
        )
    ]

    return ProjectsResponse(data=projects)


@router.get("/{project_id}", response_model=ProjectDetailResponse)
def get_project(
    *,
    project_id: str
) -> ProjectDetailResponse:

    models = [
        ModelSummary(
            model_id="9beb6f14-fe8a-4d70-8243-c51f5d7f36f8",
            name="yolox_s_test",
            type="trained_model",
            status="in_progress",
            latest_experiments=ExperimentStatus(convert="not_started", benchmark="not_started")
        ),
        ModelSummary(
            model_id="3aab6fb0-9852-4794-b668-676c06246564",
            name="yolox_l_test",
            type="compressed_model",
            status="completed",
            latest_experiments=ExperimentStatus(convert="completed", benchmark="completed")
        )
    ]

    project = ProjectDetailPayload(
        project_id=str(uuid4()),
        project_name="project_test_1",
        user_id=str(uuid4()),
        models=models
    )

    return ProjectDetailResponse(data=project)
