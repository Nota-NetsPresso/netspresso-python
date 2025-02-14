from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import api_key_header
from app.api.v1.schemas.task.benchmark.benchmark_task import SupportedDevicesResponse
from app.services.benchmark_task import benchmark_task_service
from netspresso.utils.db.session import get_db

router = APIRouter()
@router.get(
    "/benchmarks/configuration/devices",
    response_model=SupportedDevicesResponse,
    description="Get supported devices for model benchmark based on the conversion task.",
)
def get_supported_benchmark_devices(
    conversion_task_id: str = Query(..., description="Conversion task id of the model to be benchmarked."),
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> SupportedDevicesResponse:
    supported_devices = benchmark_task_service.get_supported_devices(
        db=db,
        conversion_task_id=conversion_task_id,
        api_key=api_key,
    )
    return SupportedDevicesResponse(data=supported_devices)
