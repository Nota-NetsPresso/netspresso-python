from typing import Optional
from uuid import uuid4

from fastapi import APIRouter

from app.api.v1.schemas.project import ProjectCreate, ProjectPayload, ProjectResponse, ProjectsResponse

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
