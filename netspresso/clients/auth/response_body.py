from dataclasses import dataclass


@dataclass
class TokenResponse:
    access_token: str
    refresh_token: str


@dataclass
class CreditResponse:
    free: int
    reward: int
    contract: int
    paid: int
    total: int


@dataclass
class UserDetailResponse:
    first_name: str
    last_name: str
    company: str


@dataclass
class UserResponse:
    user_id: str
    email: str
    detail_data: UserDetailResponse
    credit_info: CreditResponse
