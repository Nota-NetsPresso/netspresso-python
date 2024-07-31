from enum import Enum, IntEnum


class ServiceTask(IntEnum):
    ADVANCED_COMPRESSION = 1
    AUTOMATIC_COMPRESSION = 2
    MODEL_CONVERT = 3
    MODEL_BENCHMARK = 4


class ServiceCredit:
    CREDITS = {
        ServiceTask.ADVANCED_COMPRESSION: 50,
        ServiceTask.AUTOMATIC_COMPRESSION: 25,
        ServiceTask.MODEL_CONVERT: 50,
        ServiceTask.MODEL_BENCHMARK: 25,
    }

    @staticmethod
    def get_credit(task_id):
        return ServiceCredit.CREDITS.get(task_id, "Task not found")


class MembershipType(str, Enum):
    BASIC = "BASIC"
    PRO = "PRO"
    PREMIUM = "PREMIUM"
