from fastapi import APIRouter

from app.api.v1.schemas.system import LibraryInfo, ServerInfoPayload, ServerInfoResponse

router = APIRouter()


@router.get("/server-info", response_model=ServerInfoResponse)
def get_server_info() -> ServerInfoResponse:

    server_info = ServerInfoPayload(
        installed_libraries=[
            LibraryInfo(name="netspresso", version="1.14.0")
        ]
    )

    return ServerInfoResponse(data=server_info)
