from enum import Enum
from aenum import NamedConstant


class Task(str, Enum):
    IMAGE_CLASSIFICATION = "classification"
    OBJECT_DETECTION = "detection"
    SEMANTIC_SEGMENTATION = "segmentation"


class LauncherTask(str, Enum):
    CONVERT = "convert"
    BENCHMARK = "benchmark"


class TaskStatusForDisplay(NamedConstant):
    # task status for display - launchx
    CREATED = "IN_QUEUE"
    REQUESTED = "IN_QUEUE"
    WAITING = "IN_QUEUE"
    DOWNLOADING = "IN_PROGRESS"
    COMPILING = "IN_PROGRESS"
    RUNNING = "IN_PROGRESS"
    CONVERTED = "FINISHED"
    FINISHED = "FINISHED"
    UPLOAD_FAILED = "ERROR"
    CANCELLED = "ERROR"
    ERROR = "ERROR"
    USER_CANCEL = "USER_CANCEL"


class ConvertStatus(str, Enum):
    CREATED = "CREATED"
    REQUESTED = "REQUESTED"
    WAITING = "WAITING"
    DOWNLOADING = "DOWNLOADING"
    COMPILING = "COMPILING"
    CONVERTED = "CONVERTED"
    UPLOAD_FAILED = "UPLOAD_FAILED"
    CANCELLED = "CANCELLED"
    ERROR = "ERROR"
    USER_CANCEL = "USER_CANCEL"


class BenchmarkStatus(str, Enum):
    CREATED = "CREATED"
    REQUESTED = "REQUESTED"
    WAITING = "WAITING"
    DOWNLOADING = "DOWNLOADING"
    COMPILING = "COMPILING"
    RUNNING = "RUNNING"
    FINISHED = "ENDED"
    CANCELLED = "CANCELLED"
    ERROR = "ERROR"
    USER_CANCEL = "USER_CANCEL"
