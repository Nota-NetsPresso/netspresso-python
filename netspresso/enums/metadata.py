from enum import Enum


class TaskType(str, Enum):
    TRAIN = "train"
    COMPRESS = "compress"
    CONVERT = "convert"
    QUANTIZE = "quantize"
    BENCHMARK = "benchmark"


class Status(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"
