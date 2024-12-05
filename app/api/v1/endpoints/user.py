from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.v1.schemas.user import ApiKeyCreate, ApiKeyResponse, CreditInfo, DetailData, UserPayload, UserResponse
from app.services.user import user_service
from netspresso.utils.db.session import get_db

router = APIRouter()


@router.post("/api-key", response_model=ApiKeyResponse)
def generate_api_key(*, db: Session = Depends(get_db), request_body: ApiKeyCreate) -> ApiKeyResponse:
    api_key = user_service.generate_api_key(db=db, email=request_body.email, password=request_body.password)

    return ApiKeyResponse(data=api_key)


@router.get("/me", response_model=UserResponse)
def get_user() -> UserResponse:
    user = UserPayload(
        user_id="e8e8df79-2a62-4562-8e4d-06f51dd795b2",
        email="nppd_test_001@nota.ai",
        detail_data=DetailData(
            first_name="Byeongman",
            last_name="Lee",
            company="Nota AI",
        ),
        credit_info=CreditInfo(
            free=1000,
            total=1000,
        ),
    )

    return UserResponse(data=user)
