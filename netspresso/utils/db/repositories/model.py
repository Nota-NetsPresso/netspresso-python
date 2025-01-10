from typing import List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from netspresso.utils.db.models.model import TrainedModel
from netspresso.utils.db.repositories.base import BaseRepository, Order


class TrainedModelRepository(BaseRepository[TrainedModel]):
    def get_by_model_id(self, db: Session, model_id: str, user_id: str) -> Optional[TrainedModel]:
        model = db.query(self.model).filter(
            self.model.model_id == model_id,
            self.model.user_id == user_id,
        ).first()

        return model

    def _get_models(
        self,
        db: Session,
        condition,
        start: Optional[int] = None,
        size: Optional[int] = None,
        order: Optional[Order] = None,
    ) -> Optional[List[TrainedModel]]:
        ordering_func = self.choose_order_func(order)
        query = db.query(self.model).filter(condition)

        if order:
            query = query.order_by(ordering_func(self.model.created_at))

        if start is not None and size is not None:
            query = query.offset(start).limit(size)

        models = query.all()

        return models

    def get_all_by_user_id(
        self,
        db: Session,
        user_id: str,
        start: Optional[int] = None,
        size: Optional[int] = None,
        order: Optional[Order] = None,
    ) -> Optional[List[TrainedModel]]:
        return self._get_models(
            db=db,
            condition=self.model.user_id == user_id,
            start=start,
            size=size,
            order=order,
        )
    
    def get_all_by_project_id(
        self,
        db: Session,
        project_id: str,
        start: Optional[int] = None,
        size: Optional[int] = None,
        order: Optional[Order] = Order.DESC,
    ) -> Optional[List[TrainedModel]]:
        return self._get_models(
            db=db,
            condition=self.model.project_id == project_id,
            start=start,
            size=size,
            order=order,
        )

    def count_by_user_id(self, db: Session, user_id: str) -> int:
        return (
            db.query(func.count(self.model.user_id))
            .filter(self.model.user_id == user_id)
            .scalar()
        )


trained_model_repository = TrainedModelRepository(TrainedModel)
