from sqlalchemy import JSON, Boolean, Column, Integer, String

from netspresso.utils.db.generate_uuid import generate_uuid
from netspresso.utils.db.mixins import TimestampMixin
from netspresso.utils.db.session import Base


class Model(Base, TimestampMixin):
    __tablename__ = "model"

    id = Column(Integer, primary_key=True, index=True, unique=True, autoincrement=True, nullable=False)
    model_id = Column(String(36), index=True, unique=True, nullable=False, default=lambda: generate_uuid(entity="model"))
    name = Column(String(100), nullable=False)
    type = Column(String(30), nullable=False)
    status = Column(String(30), nullable=False)
    pretrained_model_name = Column(String(100), nullable=False)
    task = Column(String(30), nullable=False)
    framework = Column(String(30), nullable=False)
    input_shapes = Column(JSON, nullable=False)
    is_retrainable = Column(Boolean, nullable=False, default=False)
    project_id = Column(String(36), nullable=False)
    user_id = Column(String(36), nullable=False)
