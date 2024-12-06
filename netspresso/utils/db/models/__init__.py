from netspresso.utils.db.models.project import Project
from netspresso.utils.db.models.user import User
from netspresso.utils.db.session import Base, engine

Base.metadata.create_all(engine)


__all__ = [
    "Project",
    "User",
]
