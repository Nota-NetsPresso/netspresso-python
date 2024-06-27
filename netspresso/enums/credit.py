from enum import Enum, IntEnum


class ServiceCredit(IntEnum):
    ADVANCED_COMPRESSION = 50
    AUTOMATIC_COMPRESSION = 25
    MODEL_CONVERT = 50
    MODEL_BENCHMARK = 25


class MembershipType(str, Enum):
    BASIC = "BASIC"
    PRO = "PRO"
    PREMIUM = "PREMIUM"
