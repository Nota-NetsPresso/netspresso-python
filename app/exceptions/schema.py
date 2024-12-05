from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Origin(str, Enum):
    ROUTER = "router"
    SERVICE = "service"
    REPOSITORY = "repository"
    CLIENT = "client"
    LIBRARY = "library"


class AdditionalData(BaseModel):
    origin: Optional[Origin] = Field(default="", description="Error origin")
    error_log: Optional[str] = Field(default="", description="Error log")


class ExceptionDetail(BaseModel):
    data: Optional[AdditionalData] = Field(default={}, description="Additional data")
    error_code: str = Field(..., description="Error code")
    name: str = Field(..., description="Error name")
    message: str = Field(..., description="Error message")

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "data": {
                        "origin": Origin.ROUTER,
                        "error_log": "AttributeError(\"module 'np_compressor_core.torch.pruning' has no attribute 'VBMF'\")"
                    },
                    "error_code": "CS40020",
                    "name": "NotFoundMethodClassException",
                    "message": "Not found VBMF method class.",
                }
            ]
        }
