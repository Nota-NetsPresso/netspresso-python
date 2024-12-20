from typing import List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from netspresso.utils.db.models.project import Project
from netspresso.utils.db.repositories.base import BaseRepository, Order


class ProjectRepository(BaseRepository[Project]):
    def get_by_project_id(self, db: Session, project_id: str) -> Optional[Project]:
        project = db.query(self.model).filter(self.model.project_id == project_id).first()

        return project

    def _get_projects(
        self,
        db: Session,
        condition,
        start: Optional[int] = None,
        size: Optional[int] = None,
        order: Optional[Order] = None,
    ) -> Optional[List[Project]]:
        ordering_func = self.choose_order_func(order)
        query = db.query(self.model).filter(condition)

        if order:
            query = query.order_by(ordering_func(self.model.created_at))

        if start is not None and size is not None:
            query = query.offset(start).limit(size)

        projects = query.all()

        return projects

    def get_all_by_user_id(
        self,
        db: Session,
        user_id: str,
        start: Optional[int] = None,
        size: Optional[int] = None,
        order: Optional[Order] = None,
    ) -> Optional[List[Project]]:
        return self._get_projects(
            db=db,
            condition=self.model.user_id == user_id,
            start=start,
            size=size,
            order=order,
        )

    def is_project_name_duplicated(self, db: Session, project_name: str, user_id: str) -> bool:
        """
        Check if a project with the same name already exists for the given API key.

        Args:
            db (Session): Database session.
            project_name (str): The name of the project to check.
            user_id (str): The ID of the user to filter the user's projects.

        Returns:
            bool: True if the project name exists, False otherwise.
        """
        return db.query(self.model).filter(
            self.model.project_name == project_name,
            self.model.user_id == user_id,
        ).first() is not None

    def count_by_user_id(self, db: Session, user_id: str) -> int:
        return (
            db.query(func.count(self.model.user_id))
            .filter(self.model.user_id == user_id)
            .scalar()
        )


project_repository = ProjectRepository(Project)
