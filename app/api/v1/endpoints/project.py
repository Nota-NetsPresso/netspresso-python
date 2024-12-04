from uuid import uuid4

from fastapi import APIRouter

from app.api.v1.schemas.project import ProjectCreate, ProjectPayload, ProjectResponse

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
