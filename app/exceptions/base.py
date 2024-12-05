from typing import Optional

from fastapi import HTTPException

from app.exceptions.schema import AdditionalData, ExceptionDetail


class ExceptionBase(HTTPException):
    def __init__(
        self,
        data: Optional[AdditionalData],
        error_code: str,
        status_code: int,
        name: str,
        message: str,
    ):
        detail = ExceptionDetail(
            data=data,
            error_code=error_code,
            name=name,
            message=message,
        )
        super().__init__(status_code=status_code, detail=detail.model_dump())
