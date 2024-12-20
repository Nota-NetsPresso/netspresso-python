from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import api_key_header
from app.api.v1.schemas.model import ModelDetailResponse
from app.services.model import model_service
from netspresso.utils.db.session import get_db

router = APIRouter()


@router.get("/{model_id}", response_model=ModelDetailResponse)
def get_model(
    *,
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
    model_id: str
) -> ModelDetailResponse:
    model = model_service.get_model(db=db, model_id=model_id)

    return ModelDetailResponse(data=model)
