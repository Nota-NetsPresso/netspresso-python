from sqlalchemy.orm import Session

from app.services.user import user_service
from netspresso.netspresso import NetsPresso
from netspresso.utils.db.models.project import Project


class ProjectService:
    def create_project(self, db: Session, project_name: str, api_key: str) -> Project:
        user = user_service.get_user_by_api_key(db=db, api_key=api_key)

        netspresso = NetsPresso(email=user.email, password=user.password)

        project = netspresso.create_project(project_name=project_name)

        return project


project_service = ProjectService()
