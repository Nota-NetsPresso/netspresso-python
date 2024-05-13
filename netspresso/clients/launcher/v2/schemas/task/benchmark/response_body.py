import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional

from netspresso.clients.launcher.v2.schemas import (
    DeviceInfo,
    InputLayer,
    ModelOption,
    ResponseItem,
    ResponseItems,
)
from netspresso.clients.launcher.v2.schemas.task.common import TaskStatusInfo
from netspresso.enums import (
    TaskStatusForDisplay,
)
from netspresso.metadata import benchmarker
from netspresso.metadata.benchmarker import BenchmarkTaskInfo


@dataclass
class BenchmarkResult:
    processor: str
    ram_size: float
    latency: float
    power_consumption: float
    memory_footprint_cpu: float
    memory_footprint_gpu: float

    def to(self, file_size: float) -> benchmarker.BenchmarkResult:
        benchmark_result = benchmarker.BenchmarkResult()
        benchmark_result.memory_footprint_cpu = self.memory_footprint_cpu
        benchmark_result.memory_footprint_gpu = self.memory_footprint_gpu
        benchmark_result.power_consumption = self.power_consumption
        benchmark_result.ram_size = self.ram_size
        benchmark_result.latency = self.latency
        benchmark_result.file_size = file_size

        return benchmark_result


@dataclass
class BenchmarkEnvironment:
    model_framework: str
    library: list
    cpu: str = ""
    gpu: str = ""


@dataclass
class BenchmarkTask:
    benchmark_task_id: str
    input_model_id: str
    input_layer: InputLayer
    status: TaskStatusForDisplay
    benchmark_task_option: Optional[ModelOption] = None
    benchmark_result: Optional[BenchmarkResult] = None
    benchmark_environment: Optional[BenchmarkEnvironment] = None

    def __init__(self, **kwargs):
        names = {f.name for f in dataclasses.fields(self)}
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)

        self.benchmark_task_option = ModelOption(**self.benchmark_task_option)
        self.benchmark_result = BenchmarkResult(**self.benchmark_result)
        self.benchmark_environment = BenchmarkEnvironment(**self.benchmark_environment)


    def to(self) -> BenchmarkTaskInfo:
        device_info = self.benchmark_task_option.devices[0]
        benchmark_task_info = BenchmarkTaskInfo()
        benchmark_task_info.benchmark_task_uuid = self.benchmark_task_id
        benchmark_task_info.device_name = device_info.device_name
        benchmark_task_info.display_device_name = device_info.display_device_name
        benchmark_task_info.display_brand_name = device_info.display_brand_name
        benchmark_task_info.software_version = device_info.software_versions[
            0
        ].software_version
        benchmark_task_info.display_software_version = device_info.software_versions[
            0
        ].display_software_version
        benchmark_task_info.data_type = device_info.data_types[0]
        benchmark_task_info.hardware_type = (
            device_info.hardware_types[0]
            if len(device_info.hardware_types) > 0
            else None
        )

        return benchmark_task_info


@dataclass
class ResponseBenchmarkTaskItem(ResponseItem):
    data: Optional[BenchmarkTask] = field(default_factory=dict)

    def __post_init__(self):
        self.data = BenchmarkTask(**self.data)


@dataclass
class BenchmarkOption:
    option_name: str
    framework: str
    device: DeviceInfo

    def __init__(self, **kwargs):
        names = {f.name for f in dataclasses.fields(self)}
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