from typing import List, Optional

from sqlalchemy.orm import Session

from netspresso.utils.db.models.project import Project
from netspresso.utils.db.repositories.base import BaseRepository, Order


class ProjectRepository(BaseRepository[Project]):
    def get_by_project_id(self, db: Session, project_id: str) -> Optional[Project]:
        project = db.query(self.model).filter(self.model.project_id == project_id)

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


project_repository = ProjectRepository(Project)
