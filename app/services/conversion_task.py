from pathlib import Path
from typing import List

from sqlalchemy.orm import Session

from app.api.v1.schemas.task.conversion.conversion_task import (
    ConversionCreate,
    ConversionPayload,
    SupportedDeviceResponse,
    TargetFrameworkPayload,
)
from app.api.v1.schemas.task.conversion.device import (
    HardwareTypePayload,
    PrecisionPayload,
    SoftwareVersionPayload,
    SupportedDevicePayload,
    TargetDevicePayload,
)
from app.services.project import project_service
from app.services.user import user_service
from netspresso.clients.launcher.v2.schemas.common import DeviceInfo
from netspresso.enums.conversion import SourceFramework
from netspresso.utils.db.repositories.conversion import conversion_task_repository
from netspresso.utils.db.repositories.model import model_repository


class ConversionTaskService:
    def get_supported_devices(self, db: Session, framework: SourceFramework, api_key: str) -> List[SupportedDeviceResponse]:
        """Get supported devices for conversion tasks.

        Args:
            db (Session): Database session
            framework (SourceFramework): Framework to get supported devices for
            api_key (str): API key for authentication

        Returns:
            List[SupportedDeviceResponse]: List of supported devices grouped by framework
        """
        netspresso = user_service.build_netspresso_with_api_key(db=db, api_key=api_key)
        converter = netspresso.converter_v2()
        supported_options = converter.get_supported_options(framework=framework)

        return [
            self._create_supported_device_response(option)
            for option in supported_options
        ]

    def _create_supported_device_response(self, option) -> SupportedDeviceResponse:
        """Create SupportedDeviceResponse from converter option.

        Args:
            option: Converter option containing framework and devices information

        Returns:
            SupportedDeviceResponse: Response containing framework and supported devices
        """
        return SupportedDeviceResponse(
            framework=TargetFrameworkPayload(name=option.framework),
            devices=[
                self._create_device_payload(device)
                for device in option.devices
            ]
        )

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
                SoftwareVersionPayload(name=version.software_version)
                for version in device.software_versions
            ],
            precisions=[
                PrecisionPayload(name=precision)
                for precision in device.data_types
            ],
            hardware_types=[
                HardwareTypePayload(name=hardware_type)
                for hardware_type in device.hardware_types
            ],
        )

    def create_conversion_task(self, db: Session, conversion_in: ConversionCreate, api_key: str) -> ConversionPayload:
        """Create a conversion task.

        Args:
            db (Session): Database session
            conversion_in (ConversionCreate): Conversion input containing model_id and conversion settings
            api_key (str): API key for authentication

        Returns:
            ConversionResponse: Response containing conversion task
        """
        netspresso = user_service.build_netspresso_with_api_key(db=db, api_key=api_key)

        # Get model from trained models repository
        model = model_repository.get_by_model_id(db=db, model_id=conversion_in.input_model_id, user_id=netspresso.user_info.user_id)
        project = project_service.get_project(db=db, project_id=model.project_id, api_key=api_key)

        # Create output directory path as a 'converted' subfolder of input model path
        project_abs_path = Path(project.project_abs_path)
        input_model_dir = project_abs_path / model.object_path

        print(f"Input model path: {input_model_dir}")

        # Find .onnx file in the directory
        onnx_files = list(input_model_dir.glob('*.onnx'))
        if not onnx_files:
            raise FileNotFoundError(f"No .onnx file found in directory: {input_model_dir}")
        input_model_path = onnx_files[0]  # Use the first .onnx file found

        output_dir = input_model_dir / 'converted'
        output_dir.mkdir(exist_ok=True)

        print(f"Input model path: {model.object_path}")
        print(f"Output directory: {output_dir}")

        # Create conversion task
        converter = netspresso.converter_v2()
        conversion_task = converter.convert_model(
            input_model_path=str(input_model_path),
            output_dir=str(output_dir),  # Changed to use the new output_dir
            target_framework=conversion_in.framework,
            target_device_name=conversion_in.device_name,
            target_software_version=conversion_in.software_version,
            input_model_id=model.model_id,
        )

        framework = TargetFrameworkPayload(name=conversion_task.framework)
        device_name = TargetDevicePayload(name=conversion_task.device_name)
        software_version = SoftwareVersionPayload(name=conversion_task.software_version) if conversion_task.software_version else None
        precision = PrecisionPayload(name=conversion_task.precision)

        conversion_payload = ConversionPayload(
            task_id=conversion_task.task_id,
            model_id=conversion_task.model_id,
            framework=framework,
            device_name=device_name,
            software_version=software_version,
            precision=precision,
            status=conversion_task.status,
            error_detail=conversion_task.error_detail,
            input_model_id=conversion_task.input_model_id,
            created_at=conversion_task.created_at,
            updated_at=conversion_task.updated_at,
        )

        return conversion_payload

    def get_conversion_task(self, db: Session, task_id: str, api_key: str) -> ConversionPayload:
        conversion_task = conversion_task_repository.get_by_task_id(db=db, task_id=task_id)

        framework = TargetFrameworkPayload(name=conversion_task.framework)
        device_name = TargetDevicePayload(name=conversion_task.device_name)
        software_version = SoftwareVersionPayload(name=conversion_task.software_version) if conversion_task.software_version else None
        precision = PrecisionPayload(name=conversion_task.precision)

        conversion_payload = ConversionPayload(
            task_id=conversion_task.task_id,
            model_id=conversion_task.model_id,
            framework=framework,
            device_name=device_name,
            software_version=software_version,
            precision=precision,
            status=conversion_task.status,
            error_detail=conversion_task.error_detail,
            input_model_id=conversion_task.input_model_id,
            created_at=conversion_task.created_at,
            updated_at=conversion_task.updated_at,
        )

        return conversion_payload


conversion_task_service = ConversionTaskService()
