from pathlib import Path
from typing import List

from sqlalchemy.orm import Session

from app.api.v1.schemas.device import (
    HardwareTypePayload,
    PrecisionPayload,
    SoftwareVersionPayload,
    SupportedDevicePayload,
    SupportedDeviceResponse,
)
from app.api.v1.schemas.task.benchmark.benchmark_task import (
    BenchmarkCreate,
    BenchmarkCreatePayload,
    TargetFrameworkPayload,
)
from app.services.conversion_task import conversion_task_service
from app.services.project import project_service
from app.services.user import user_service
from app.worker.celery_app import benchmark_model_task
from netspresso.clients.launcher.v2.schemas.common import DeviceInfo
from netspresso.utils.db.repositories.model import model_repository


class BenchmarkTaskService:
    def get_supported_devices(
        self, db: Session, conversion_task_id: str, api_key: str
    ) -> List[SupportedDeviceResponse]:
        """Get supported devices for conversion tasks.

        Args:
            db (Session): Database session
            conversion_task_id (str): Conversion task ID
            api_key (str): API key for authentication

        Returns:
            List[SupportedDeviceResponse]: List of supported devices grouped by framework
        """
        netspresso = user_service.build_netspresso_with_api_key(db=db, api_key=api_key)
        benchmarker = netspresso.benchmarker_v2()

        conversion_task = conversion_task_service.get_conversion_task(
            db=db, task_id=conversion_task_id, api_key=api_key
        )

        framework = conversion_task.framework.name
        device = conversion_task.device.name
        software_version = conversion_task.software_version.name if conversion_task.software_version else None

        supported_options = benchmarker.get_supported_options(
            framework=framework, device=device, software_version=software_version
        )

        return [self._create_supported_device_response(option) for option in supported_options]

    def _create_supported_device_response(self, option) -> SupportedDeviceResponse:
        """Create SupportedDeviceResponse from converter option.

        Args:
            option: Converter option containing framework and devices information

        Returns:
            SupportedDeviceResponse: Response containing framework and supported devices
        """
        response = SupportedDeviceResponse(
            framework=TargetFrameworkPayload(name=option.framework),
            devices=[self._create_device_payload(device) for device in option.devices],
        )

        return response

    def _create_device_payload(self, device: DeviceInfo) -> SupportedDevicePayload:
        """Create SupportedDevicePayload from device information.

        Args:
            device: Device information containing name, versions, precisions, and hardware types

        Returns:
            SupportedDevicePayload: Payload containing device information
        """
        return SupportedDevicePayload(
            name=device.device_name,
            software_versions=[
                SoftwareVersionPayload(name=version.software_version) for version in device.software_versions
            ],
            precisions=[PrecisionPayload(name=precision) for precision in device.data_types],
            hardware_types=[HardwareTypePayload(name=hardware_type) for hardware_type in device.hardware_types],
        )

    def create_benchmark_task(self, db: Session, benchmark_in: BenchmarkCreate, api_key: str) -> BenchmarkCreatePayload:
        netspresso = user_service.build_netspresso_with_api_key(db=db, api_key=api_key)

        # Get model from trained models repository
        model = model_repository.get_by_model_id(
            db=db, model_id=benchmark_in.input_model_id, user_id=netspresso.user_info.user_id
        )
        project = project_service.get_project(db=db, project_id=model.project_id, api_key=api_key)

        # Create output directory path as a 'converted' subfolder of input model path
        project_abs_path = Path(project.project_abs_path)
        input_model_path = project_abs_path / model.object_path

        print(f"Input model path: {input_model_path}")

        task = benchmark_model_task.delay(
            api_key=api_key,
            input_model_path=input_model_path.as_posix(),
            target_device_name=benchmark_in.device_name,
            target_software_version=benchmark_in.software_version,
            target_hardware_type=benchmark_in.hardware_type,
            input_model_id=benchmark_in.input_model_id,
        )
        task_id = task.get()
        return BenchmarkCreatePayload(task_id=task_id)


benchmark_task_service = BenchmarkTaskService()
