from sqlalchemy import Boolean, Column, Integer, String

from netspresso.utils.db.mixins import TimestampMixin
from netspresso.utils.db.session import Base


class User(Base, TimestampMixin):
    __tablename__ = "user"

    id = Column(Integer, primary_key=True, index=True, unique=True, autoincrement=True, nullable=False)
    email = Column(String(36), nullable=False)
    password = Column(String(36), nullable=False)
    api_key = Column(String(36), nullable=False)
    user_id = Column(String(36), nullable=False)
    is_deleted = Column(Boolean, nullable=False, default=False)
