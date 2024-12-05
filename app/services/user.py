from sqlalchemy.orm import Session

from app.api.v1.schemas.user import ApiKeyPayload
from app.utils import generate_id, hash_password
from netspresso.utils.db.models.user import User
from netspresso.utils.db.repositories.user import user_repository


class UserService:
    def create_user(self, db: Session, email: str, password: str, api_key: str):
        hashed_password = hash_password(password)

        user = User(
            email=email,
            password=hashed_password,
            api_key=api_key,
        )
        user = user_repository.save(db=db, model=user)

        return user

    def generate_api_key(self, db: Session, email: str, password: str) -> ApiKeyPayload:
        generated_id = generate_id(entity="user")

        user = user_repository.get_by_email(db=db, email=email)

        if user:
            user.api_key = generated_id
            user.password = hash_password(password)
            user_repository.save(db=db, model=user)
        else:
            user = self.create_user(
                db=db,
                email=email,
                password=password,
                api_key=generated_id,
            )

        api_key = ApiKeyPayload(api_key=generated_id)

        return api_key


user_service = UserService()
