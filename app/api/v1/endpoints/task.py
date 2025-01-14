from typing import Any, Dict, List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import api_key_header
from app.api.v1.schemas.task.train.train_task import TrainTaskDetailResponse
from app.services.task import task_service
from netspresso.utils.db.session import get_db

router = APIRouter()


@router.get("/train/configuration/models", response_model=Dict[str, List[str]], description="Get supported models for training tasks.")
def get_supported_models(
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> Dict[str, List[str]]:
    supported_models = task_service.get_supported_models(db=db, api_key=api_key)

    return supported_models


@router.get("/train/configuration/optimizers", response_model=List[Dict[str, Any]], description="Get supported optimizers for training tasks.")
def get_supported_optimizers(
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> List[Dict[str, Any]]:
    supported_optimizers = task_service.get_supported_optimizers(db=db, api_key=api_key)

    return supported_optimizers


@router.get("/train/configuration/schedulers", response_model=List[Dict[str, Any]], description="Get supported schedulers for training tasks.")
def get_supported_schedulers(
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> List[Dict[str, Any]]:
    supported_schedulers = task_service.get_supported_schedulers(db=db, api_key=api_key)

    return supported_schedulers


@router.get("/tasks/train/{task_id}", response_model=TrainTaskDetailResponse)
def get_task(
    *,
    task_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> TrainTaskDetailResponse:
    task = task_service.get_task(db=db, task_id=task_id, api_key=api_key)

    return TrainTaskDetailResponse(data=task)
