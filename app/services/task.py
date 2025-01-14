from typing import Dict, List

from sqlalchemy.orm import Session

from app.api.v1.schemas.task.train.train_task import TrainTaskSchema
from app.services.user import user_service
from netspresso.utils.db.repositories.task import train_task_repository


class TaskService:
    def get_supported_models(self, db: Session, api_key: str) -> Dict[str, List[str]]:
        netspresso = user_service.build_netspresso_with_api_key(db=db, api_key=api_key)

        trainer = netspresso.trainer()

        return trainer.get_all_available_models()

    def get_task(self, db: Session, task_id: str, api_key: str) -> TrainTaskSchema:
        netspresso = user_service.build_netspresso_with_api_key(db=db, api_key=api_key)

        train_task = train_task_repository.get_by_task_id(db=db, task_id=task_id)
        train_task = TrainTaskSchema.model_validate(train_task)

        return train_task


task_service = TaskService()
