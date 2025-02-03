from typing import List

from sqlalchemy.orm import Session

from app.api.v1.schemas.task.conversion.conversion_task import FrameworkPayload, SupportedDeviceResponse
from app.api.v1.schemas.task.conversion.device import (
    DevicePayload,
    HardwareTypePayload,
    PrecisionPayload,
    SoftwareVersionPayload,
)
from app.services.user import user_service
from netspresso.clients.launcher.v2.schemas.common import DeviceInfo
from netspresso.enums.conversion import Framework


class ConversionTaskService:
    def get_supported_devices(self, db: Session, framework: Framework, api_key: str) -> List[SupportedDeviceResponse]:
        """Get supported devices for conversion tasks.

        Args:
            db (Session): Database session
            framework (Framework): Framework to get supported devices for
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
            framework=FrameworkPayload(name=option.framework),
            devices=[
                self._create_device_payload(device)
                for device in option.devices
            ]
        )

    def _create_device_payload(self, device: DeviceInfo) -> DevicePayload:
        """Create DevicePayload from device information.

        Args:
            device: Device information containing name, versions, precisions, and hardware types

        Returns:
            DevicePayload: Payload containing device information
        """
        return DevicePayload(
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


conversion_task_service = ConversionTaskService()
