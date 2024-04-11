import dataclasses
from dataclasses import dataclass, field
from typing import Optional, List

from netspresso.enums import (
    TaskStatusForDisplay,
)
from netspresso.clients.launcher.v2.schemas.task.common import TaskStatusInfo
from netspresso.clients.launcher.v2.schemas import (
    ResponseItem,
    ResponseItems,
    InputLayer,
    DeviceInfo,
    ModelOption,
)


@dataclass(init=False)
class BenchmarkResult:
    processor: str
    ram_size: float
    latency: float
    power_consumption: float
    memory_footprint_cpu: float
    memory_footprint_gpu: float


@dataclass(init=False)
class BenchmarkEnvironment:
    model_framework: str
    library: list
    cpu: str = ""
    gpu: str = ""


@dataclass(init=False)
class BenchmarkTask:
    benchmark_task_id: str
    input_model_id: str
    input_layer: InputLayer
    status: TaskStatusForDisplay
    benchmark_task_option: Optional[ModelOption] = None
    benchmark_result: Optional[BenchmarkResult] = None
    benchmark_environment: Optional[BenchmarkEnvironment] = None

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)


@dataclass
class ResponseBenchmarkTaskItem(ResponseItem):
    data: Optional[BenchmarkTask] = field(default_factory=dict)

    def __post_init__(self):
        self.data = BenchmarkTask(**self.data)


@dataclass(init=False)
class BenchmarkOption:
    option_name: str
    framework: str
    device: DeviceInfo

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)


@dataclass
class ResponseBenchmarkOptionItems(ResponseItems):
    data: List[Optional[BenchmarkOption]] = field(default_factory=list)

    def __post_init__(self):
        self.data = [BenchmarkOption(**item) for item in self.data]


@dataclass
class ResponseBenchmarkStatusItem(ResponseItem):
    data: TaskStatusInfo = field(default_factory=TaskStatusInfo)

    def __post_init__(self):
        self.data = TaskStatusInfo(**self.data)
