from sqlalchemy.orm import Session

from app.api.v1.schemas.user import ApiKeyPayload
from app.utils import generate_id
from netspresso.utils.db.models.user import User
from netspresso.utils.db.repositories.user import user_repository


class UserService:
    def create_user(self, db: Session, email: str, password: str, api_key: str):
        user = User(
            email=email,
            password=password,
            api_key=api_key,
        )
        user = user_repository.save(db=db, model=user)

        return user

    def generate_api_key(self, db: Session, email: str, password: str) -> ApiKeyPayload:
        generated_id = generate_id(entity="user")

        user = user_repository.get_by_email(db=db, email=email)

        if user:
            if user.password != password:
                user.password = password
                user.api_key = generated_id
            elif user.api_key != generated_id:
                user.api_key = generated_id
            user = user_repository.save(db=db, model=user)
        else:
            user = self.create_user(
                db=db,
                email=email,
                password=password,
                api_key=generated_id,
            )

        api_key = ApiKeyPayload(api_key=user.api_key)

        return api_key


user_service = UserService()
