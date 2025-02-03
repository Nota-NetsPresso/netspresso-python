from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

from app.api.v1.schemas.task.conversion.device import DevicePayload
from netspresso.enums.conversion import FRAMEWORK_DISPLAY_MAP, Framework, FrameworkDisplay


class FrameworkPayload(BaseModel):
    name: Framework = Field(description="Framework name")
    display_name: Optional[FrameworkDisplay] = Field(default=None, description="Framework display name")

    @model_validator(mode="after")
    def set_display_name(self) -> str:
        self.display_name = FRAMEWORK_DISPLAY_MAP.get(self.name)

        return self


class SupportedDeviceResponse(BaseModel):
    framework: FrameworkPayload
    devices: List[DevicePayload]


class SupportedDevicesResponse(BaseModel):
    data: List[SupportedDeviceResponse]


class SupportedDeviceRequest(BaseModel):
    framework: Framework


class ConversionCreate(BaseModel):
    pass


class ConversionResponse(BaseModel):
    pass

