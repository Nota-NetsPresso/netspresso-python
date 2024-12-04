import warnings
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")


class Order(Enum):
    DESC = "desc"
    ASC = "asc"


class BaseResponse(BaseModel):
    code: int
    message: str


class Paging(BaseModel):
    start: int = Field(description="list items offset", default=0)
    size: int = Field(description="list items size", default=10)
    order: Order = Field(description="list items order option", default="desc")
    field_name: str = Field(description="list items order field name", default="created_at")


class ResponsePaginationItems(BaseResponse):
    data: List[dict] = Field(description="list items")
    result_count: int = Field(description="return items count", default=0)
    total_count: int = Field(description="total items count", default=0)


class ResponseItem(BaseResponse):
    data: Optional[dict]


class ResponseListItems(ResponsePaginationItems):
    data: List[dict]
