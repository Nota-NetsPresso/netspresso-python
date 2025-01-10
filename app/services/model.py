from typing import List
from sqlalchemy.orm import Session

from app.services.user import user_service
from app.api.v1.schemas.model import ModelPayload
from netspresso.utils.db.repositories.model import trained_model_repository


class ModelService:
    def get_models(self, db: Session, api_key: str) -> List[ModelPayload]:
        netspresso = user_service.build_netspresso_with_api_key(db=db, api_key=api_key)

        models = trained_model_repository.get_all_by_user_id(db=db, user_id=netspresso.user_info.user_id)
        models = [ModelPayload.model_validate(model) for model in models]

        return models

    def get_model(self, db: Session, model_id: str, api_key: str) -> ModelPayload:
        netspresso = user_service.build_netspresso_with_api_key(db=db, api_key=api_key)

        model = trained_model_repository.get_by_model_id(db=db, model_id=model_id, user_id=netspresso.user_info.user_id)
        model = ModelPayload.model_validate(model)

        return model


model_service = ModelService()
