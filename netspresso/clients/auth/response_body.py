from pydantic import BaseModel, EmailStr, Field


class TokenResponse(BaseModel):
    access_token: str = Field(..., description="Access Token")
    refresh_token: str = Field(..., description="Refresh Token")


class CreditResponse(BaseModel):
    free: int = Field(..., description="Free Credit")
    reward: int = Field(..., description="Reward Credit")
    contract: int = Field(..., description="Contract Credit")
    paid: int = Field(..., description="Paid Credit")
    total: int = Field(..., description="Total Credit")


class UserDetailResponse(BaseModel):
    first_name: str = Field(..., description="First Name")
    last_name: str = Field(..., description="Last Name")
    company: str = Field(..., description="Company")


class UserResponse(BaseModel):
    user_id: str = Field(..., description="User ID")
    email: EmailStr = Field(..., description="Email")
    detail_data: UserDetailResponse = Field(..., description="User Detail")
    credit_info: CreditResponse = Field(..., description="Credit Info")