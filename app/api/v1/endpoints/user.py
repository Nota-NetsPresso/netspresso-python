from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import api_key_header
from app.api.v1.schemas.user import ApiKeyCreate, ApiKeyResponse, UserResponse
from app.services.user import user_service
from netspresso.utils.db.session import get_db

router = APIRouter()


@router.post("/api-key", response_model=ApiKeyResponse)
def generate_api_key(*, db: Session = Depends(get_db), request_body: ApiKeyCreate) -> ApiKeyResponse:
    api_key = user_service.generate_api_key(db=db, email=request_body.email, password=request_body.password)

    return ApiKeyResponse(data=api_key)


@router.get("/me", response_model=UserResponse)
def get_user(*, db: Session = Depends(get_db), api_key: str = Depends(api_key_header)) -> UserResponse:
    user = user_service.get_user_info(db=db, api_key=api_key)

    return UserResponse(data=user)
