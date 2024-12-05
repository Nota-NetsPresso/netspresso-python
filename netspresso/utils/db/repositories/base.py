from enum import Enum
from typing import Generic, Type, TypeVar

from sqlalchemy import asc, desc
from sqlalchemy.orm import Session

from netspresso.utils.db.session import Base

ModelType = TypeVar("ModelType", bound=Base) # type: ignore


class Order(str, Enum):
    DESC = "desc"
    ASC = "asc"


class BaseRepository(Generic[ModelType]):
    def __init__(self, model: Type[ModelType]):
        self.model = model

    def choose_order_func(self, order):
        if order == Order.DESC:
            return desc
        return asc

    def save(self, db: Session, model: ModelType) -> ModelType:
        db.add(model)
        db.commit()
        db.refresh(model)

        return model

    def update(self, db: Session, model: ModelType) -> ModelType:
        return self.save(db, model)
