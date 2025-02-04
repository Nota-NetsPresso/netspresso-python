from typing import List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from netspresso.utils.db.models.conversion import ConversionTask
from netspresso.utils.db.repositories.base import BaseRepository, Order


class ConversionTaskRepository(BaseRepository[ConversionTask]):
    def get_by_task_id(self, db: Session, task_id: str) -> Optional[ConversionTask]:
        task = db.query(self.model).filter(
            self.model.task_id == task_id,
        ).first()

        return task

    def save(self, db, task):
        db.add(task)
        db.commit()
        db.refresh(task)

        return task

conversion_task_repository = ConversionTaskRepository(ConversionTask)
