from fastapi import APIRouter

from app.api.v1.schemas.system import GpusInfoResponse, ServerInfoPayload, ServerInfoResponse
from app.services.system import system_service

router = APIRouter()


@router.get("/server-info", response_model=ServerInfoResponse)
def get_server_info() -> ServerInfoResponse:
    installed_libraries = system_service.get_installed_libraries()

    server_info = ServerInfoPayload(installed_libraries=installed_libraries)

    return ServerInfoResponse(data=server_info)


@router.get("/gpus-info", response_model=GpusInfoResponse)
def get_gpus_info() -> GpusInfoResponse:

    gpus_info = system_service.get_gpus_info()

    return GpusInfoResponse(data=gpus_info)
