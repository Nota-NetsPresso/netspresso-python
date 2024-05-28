from dataclasses import dataclass, field

from netspresso.clients.auth.response_body import CreditResponse
from netspresso.clients.auth.v2.schemas.common import AbstractResponse


@dataclass
class CreditWithType:
    user_id: str
    free_credit: int
    paid_credit: int
    total_credit: int


@dataclass
class SummarizedCreditResponse(AbstractResponse):
    data: CreditWithType = field(default_factory=CreditWithType)

    def __post_init__(self):
        self.data = CreditWithType(**self.data)

    def to(self) -> CreditResponse:
        return CreditResponse(
            free=self.data.free_credit,
            reward=0,
            contract=0,
            paid=self.data.paid_credit,
            total=self.data.total_credit,
        )
