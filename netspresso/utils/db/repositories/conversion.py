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

    def _get_tasks(
        self,
        db: Session,
        condition,
        start: Optional[int] = None,
        size: Optional[int] = None,
        order: Optional[Order] = Order.DESC,
    ) -> Optional[List[ConversionTask]]:
        ordering_func = self.choose_order_func(order)
        query = db.query(self.model).filter(condition)

        if order:
            query = query.order_by(ordering_func(self.model.updated_at))

        if start is not None and size is not None:
            query = query.offset(start).limit(size)

        models = query.all()

        return models

    def get_all_by_model_id(
        self,
        db: Session,
        model_id: str,
        start: Optional[int] = None,
        size: Optional[int] = None,
        order: Optional[Order] = Order.DESC,
    ) -> Optional[List[ConversionTask]]:
        return self._get_tasks(
            db=db,
            condition=self.model.input_model_id == model_id,
            start=start,
            size=size,
            order=order,
        )

conversion_task_repository = ConversionTaskRepository(ConversionTask)
