from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import api_key_header
from app.api.v1.schemas.train_task import TrainTaskDetailResponse
from app.services.task import task_service
from netspresso.utils.db.session import get_db

router = APIRouter()


@router.get("/tasks/train/{task_id}", response_model=TrainTaskDetailResponse)
def get_task(
    *,
    task_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> TrainTaskDetailResponse:
    task = task_service.get_task(db=db, task_id=task_id, api_key=api_key)

    return TrainTaskDetailResponse(data=task)
