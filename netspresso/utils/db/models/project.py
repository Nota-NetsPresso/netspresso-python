from sqlalchemy import Column, Integer, String

from netspresso.utils.db.generate_uuid import generate_uuid
from netspresso.utils.db.mixins import TimestampMixin
from netspresso.utils.db.session import Base


class Project(Base, TimestampMixin):
    __tablename__ = "project"

    id = Column(Integer, primary_key=True, index=True, unique=True, autoincrement=True, nullable=False)
    project_id = Column(String(36), index=True, unique=True, nullable=False, default=lambda: generate_uuid(entity="project"))
    project_name = Column(String(30), nullable=False, unique=True)
    user_id = Column(String(36), nullable=False)
    project_abs_path = Column(String(500), nullable=False)
