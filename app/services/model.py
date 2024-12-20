from sqlalchemy.orm import Session

from app.api.v1.schemas.model import ModelDetailPayload
from netspresso.utils.db.repositories.model import trained_model_repository


class ModelService:
    def get_model(self, db: Session, model_id: str) -> ModelDetailPayload:
        model = trained_model_repository.get_by_model_id(db=db, model_id=model_id)
        model = ModelDetailPayload.model_validate(model)
        model.status = model.train_task.status

        return model


model_service = ModelService()
