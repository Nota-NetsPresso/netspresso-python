from typing import List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from netspresso.utils.db.models.train import TrainTask
from netspresso.utils.db.repositories.base import BaseRepository, Order


class TrainTaskRepository(BaseRepository[TrainTask]):
    def get_by_task_id(self, db: Session, task_id: str) -> Optional[TrainTask]:
        task = db.query(self.model).filter(
            self.model.task_id == task_id,
        ).first()

        return task

    def save(self, db, task):
        db.add(task)
        db.commit()
        db.refresh(task)

        return task

train_task_repository = TrainTaskRepository(TrainTask)
