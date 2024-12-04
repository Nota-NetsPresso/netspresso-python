from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from app.api.v1.schemas.base import ResponseItem


class LibraryInfo(BaseModel):
    name: str = Field(..., description="The name of the installed library. Example: 'netspresso'.")
    version: str = Field(..., description="The version of the installed library. Example: '1.14.0'.")


class ServerInfoPayload(BaseModel):
    installed_libraries: List[LibraryInfo] = Field(..., description="A list of installed libraries on the server.")


class GpuInfoPayload(BaseModel):
    index: int = Field(..., description="The index of the GPU within the system.")
    uuid: str = Field(..., description="The unique identifier (UUID) of the GPU.")
    name: str = Field(..., description="The name of the GPU (e.g., NVIDIA GeForce RTX 3070).")
    temperature_gpu: Optional[int] = Field(None, description="The current temperature of the GPU in Celsius.")
    fan_speed: Optional[int] = Field(None, description="The fan speed of the GPU, represented as a percentage.")
    utilization_gpu: Optional[int] = Field(None, description="The GPU utilization rate as a percentage.")
    utilization_enc: Optional[int] = Field(None, description="The GPU encoder utilization rate as a percentage.")
    utilization_dec: Optional[int] = Field(None, description="The GPU decoder utilization rate as a percentage.")
    power_draw: Optional[int] = Field(None, description="The current power consumption of the GPU in watts.")
    enforced_power_limit: Optional[int] = Field(None, description="The (enforced) GPU power limit in Watts,")
    memory_used: int = Field(..., description="The amount of GPU memory currently in use, measured in MiB.")
    memory_total: int = Field(..., description="The total amount of GPU memory available, measured in MiB.")
    processes: Optional[List[Dict]] = Field(default_factory=list, description="A list of processes running on the GPU.")


class ServerInfoResponse(ResponseItem):
    data: ServerInfoPayload


class GpuInfosResponse(ResponseItem):
    data: List[GpuInfoPayload]
