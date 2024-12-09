from typing import Optional

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import api_key_header
from app.api.v1.schemas.base import Order
from app.api.v1.schemas.project import (
    ExperimentStatus,
    ModelSummary,
    ProjectCreate,
    ProjectDetailPayload,
    ProjectDetailResponse,
    ProjectDuplicationCheckResponse,
    ProjectDuplicationStatus,
    ProjectResponse,
    ProjectsResponse,
    ProjectSummaryPayload,
)
from app.services.project import project_service
from netspresso.utils.db.session import get_db

router = APIRouter()


@router.post("", response_model=ProjectResponse)
def create_project(
    *,
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
    request_body: ProjectCreate,
) -> ProjectResponse:
    project = project_service.create_project(db=db, project_name=request_body.project_name, api_key=api_key)

    project = ProjectSummaryPayload.model_validate(project)

    return ProjectResponse(data=project)


@router.post("/duplicate", response_model=ProjectDuplicationCheckResponse)
def check_project_duplication(
    *,
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
    request_body: ProjectCreate,
) -> ProjectDuplicationCheckResponse:
    is_duplicated = project_service.check_project_duplication(
        db=db, project_name=request_body.project_name, api_key=api_key
    )

    duplication_status = ProjectDuplicationStatus(is_duplicated=is_duplicated)

    return ProjectDuplicationCheckResponse(data=duplication_status)


@router.get("", response_model=ProjectsResponse)
def get_projects(
    *,
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
    start: Optional[int] = 0,
    size: Optional[int] = 10,
    order: Order = Order.DESC,
) -> ProjectsResponse:
    projects = project_service.get_projects(db=db, start=start, size=size, order=order, api_key=api_key)
    projects = [ProjectSummaryPayload.model_validate(project) for project in projects]
    total_count = project_service.count_project_by_user_id(db=db, api_key=api_key)

    return ProjectsResponse(data=projects, result_count=len(projects), total_count=total_count)


@router.get("/{project_id}", response_model=ProjectDetailResponse)
def get_project(
    *, project_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> ProjectDetailResponse:
    project_summary = project_service.get_project(db=db, project_id=project_id, api_key=api_key)
    project_summary_payload = ProjectSummaryPayload.model_validate(project_summary)

    # TODO: Add get_model_summary
    models = [
        ModelSummary(
            model_id="9beb6f14-fe8a-4d70-8243-c51f5d7f36f8",
            name="yolox_s_test",
            type="trained_model",
            status="in_progress",
            latest_experiments=ExperimentStatus(convert="not_started", benchmark="not_started"),
        ),
        ModelSummary(
            model_id="3aab6fb0-9852-4794-b668-676c06246564",
            name="yolox_l_test",
            type="compressed_model",
            status="completed",
            latest_experiments=ExperimentStatus(convert="completed", benchmark="completed"),
        ),
    ]

    project_detail = ProjectDetailPayload(**project_summary_payload.model_dump(), models=models)

    return ProjectDetailResponse(data=project_detail)
