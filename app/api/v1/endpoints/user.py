from fastapi import APIRouter

from app.api.v1.schemas.user import CreditInfo, DetailData, UserPayload, UserResponse

router = APIRouter()


@router.post("/me", response_model=UserResponse)
def get_user() -> UserResponse:

    project = UserPayload(
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

    return UserResponse(data=project)
