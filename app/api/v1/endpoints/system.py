from fastapi import APIRouter
from gpustat import GPUStatCollection

from app.api.v1.schemas.system import (
    GpuInfoPayload,
    GpuInfosResponse,
    LibraryInfo,
    ServerInfoPayload,
    ServerInfoResponse,
)

router = APIRouter()


@router.get("/server-info", response_model=ServerInfoResponse)
def get_server_info() -> ServerInfoResponse:

    server_info = ServerInfoPayload(installed_libraries=[LibraryInfo(name="netspresso", version="1.14.0")])

    return ServerInfoResponse(data=server_info)


@router.get("/gpu-info", response_model=GpuInfosResponse)
def get_gpu_infos() -> GpuInfosResponse:

    stats = GPUStatCollection.new_query()

    gpu_infos = [
        GpuInfoPayload(
            index=gpu.index,
            uuid=gpu.uuid,
            name=gpu.name,
            temperature_gpu=gpu.temperature,
            fan_speed=gpu.fan_speed,
            utilization_gpu=gpu.utilization,
            utilization_enc=gpu.utilization_enc,
            utilization_dec=gpu.utilization_dec,
            power_draw=gpu.power_draw,
            enforced_power_limit=gpu.power_limit,
            memory_used=gpu.memory_used,
            memory_total=gpu.memory_total,
            processes=list(gpu.processes),
        )
        for gpu in stats.gpus
    ]

    return GpuInfosResponse(data=gpu_infos)
