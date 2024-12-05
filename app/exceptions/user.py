from fastapi import status

from app.exceptions.base import ExceptionBase
from app.exceptions.schema import AdditionalData, Origin


class IncorrectEmailOrPasswordException(ExceptionBase):
    def __init__(self, origin: Origin = Origin.SERVICE):
        super().__init__(
            data=AdditionalData(origin=origin),
            error_code="US40101",
            status_code=status.HTTP_401_UNAUTHORIZED,
            name=self.__class__.__name__,
            message="The email or password provided is incorrect. Please check your email and password and try again.",
        )
