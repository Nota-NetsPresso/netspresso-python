from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from netspresso.utils.db.generate_uuid import generate_uuid
from netspresso.utils.db.mixins import TimestampMixin
from netspresso.utils.db.session import Base


class TrainedModel(Base, TimestampMixin):
    __tablename__ = "trained_model"

    id = Column(Integer, primary_key=True, index=True, unique=True, autoincrement=True, nullable=False)
    model_id = Column(String(36), index=True, unique=True, nullable=False, default=lambda: generate_uuid(entity="model"))
    name = Column(String(100), nullable=False)
    type = Column(String(30), nullable=False)
    is_retrainable = Column(Boolean, nullable=False, default=False)
    project_id = Column(String(36), nullable=False)
    user_id = Column(String(36), nullable=False)
    train_task = relationship("TrainTask", back_populates="model", uselist=False, cascade="all, delete-orphan")
