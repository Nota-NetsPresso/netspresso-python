from enum import Enum
from dataclasses import dataclass


@dataclass
class AbstractResponse:
    pass


@dataclass
class PagingResponse(AbstractResponse):
    total_count: int
    result_count: int


class MembershipType(str, Enum):
    BASIC = "BASIC"
    PRO = "PRO"
    PREMIUM = "PREMIUM"


class CreditType(str, Enum):
    FREE = "FREE"
    PAID = "PAID"
