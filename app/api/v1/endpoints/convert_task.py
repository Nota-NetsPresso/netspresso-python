from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import api_key_header
from app.api.v1.schemas.task.conversion.conversion_task import (
    SupportedDevicesResponse,
)
from app.services.conversion_task import conversion_task_service
from netspresso.enums.conversion import Framework
from netspresso.utils.db.session import get_db

router = APIRouter()


@router.get(
    "/conversions/configurations/devices",
    response_model=SupportedDevicesResponse,
    description="Get supported devices for conversion tasks",
)
def get_supported_conversion_devices(
    framework: Framework = Query(..., description="Framework name to get supported devices"),
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> SupportedDevicesResponse:
    supported_devices = conversion_task_service.get_supported_devices(db=db, framework=framework, api_key=api_key)

    return SupportedDevicesResponse(data=supported_devices)
