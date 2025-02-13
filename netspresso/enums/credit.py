from enum import Enum


class ServiceTask(str, Enum):
    TRAINING = "Training"
    ADVANCED_COMPRESSION = "Advanced Compression"
    AUTOMATIC_COMPRESSION = "Automatic Compression"
    MODEL_CONVERT = "Conversion"
    MODEL_BENCHMARK = "Benchmark"
    MODEL_QUANTIZE = "Quantization"


class ServiceCredit:
    CREDITS = {
        ServiceTask.ADVANCED_COMPRESSION: 50,
        ServiceTask.AUTOMATIC_COMPRESSION: 25,
        ServiceTask.MODEL_CONVERT: 50,
        ServiceTask.MODEL_BENCHMARK: 25,
        ServiceTask.MODEL_QUANTIZE: 50,
    }

    @staticmethod
    def get_credit(task_id):
        return ServiceCredit.CREDITS.get(task_id, "Task not found")


class MembershipType(str, Enum):
    BASIC = "BASIC"
    PRO = "PRO"
    PREMIUM = "PREMIUM"
