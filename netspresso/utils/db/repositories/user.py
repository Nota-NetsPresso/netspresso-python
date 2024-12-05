from typing import Optional

from sqlalchemy.orm import Session

from netspresso.utils.db.models.user import User
from netspresso.utils.db.repositories.base import BaseRepository


class UserRepository(BaseRepository[User]):
    def get_by_email(self, db: Session, email: str) -> Optional[User]:
        user = db.query(User).filter(User.email == email).first()

        return user

    def get_by_user_id(self, db: Session, user_id: str) -> Optional[User]:
        user = db.query(self.model).filter(self.model.user_id == user_id).first()

        return user

    def get_by_api_key(self, db: Session, api_key: str) -> Optional[User]:
        user = db.query(self.model).filter(self.model.api_key == api_key).first()

        return user


user_repository = UserRepository(User)
