from typing import List

from pydantic import BaseModel, Field

from app.api.v1.schemas.base import ResponseItem


class LibraryInfo(BaseModel):
    name: str = Field(..., description="The name of the installed library. Example: 'netspresso'.")
    version: str = Field(..., description="The version of the installed library. Example: '1.14.0'.")


class ServerInfoPayload(BaseModel):
    installed_libraries: List[LibraryInfo] = Field(..., description="A list of installed libraries on the server.")


class ServerInfoResponse(ResponseItem):
    data: ServerInfoPayload
