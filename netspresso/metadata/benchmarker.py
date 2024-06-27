from dataclasses import dataclass, field
from typing import List

from netspresso.enums import (
    DataType,
    DeviceName,
    HardwareType,
    SoftwareVersion,
    Status,
    TaskType,
)


@dataclass
class BenchmarkTaskInfo:
    benchmark_task_uuid: str = ""
    device_name: DeviceName = ""
    display_device_name: str = ""
    display_brand_name: str = ""
    software_version: SoftwareVersion = ""
    display_software_version: str = ""
    data_type: DataType = ""
    hardware_type: HardwareType = ""


@dataclass
class BenchmarkResult:
    memory_footprint_gpu: int = None
    memory_footprint_cpu: int = None
    power_consumption: int = None
    ram_size: int = None
    latency: int = None
    file_size: int = None


@dataclass
class BenchmarkEnvironment:
    model_framework: str = ""
    system: str = ""
    machine: str = ""
    cpu: str = ""
    gpu: str = ""
    library: List[str] = field(default_factory=list)


@dataclass
class BenchmarkerMetadata:
    status: Status = Status.IN_PROGRESS
    message: str = ""
    task_type: TaskType = TaskType.BENCHMARK
    input_model_path: str = ""
    benchmark_task_info: BenchmarkTaskInfo = field(default_factory=BenchmarkTaskInfo)
    benchmark_result: BenchmarkResult = field(default_factory=BenchmarkResult)
