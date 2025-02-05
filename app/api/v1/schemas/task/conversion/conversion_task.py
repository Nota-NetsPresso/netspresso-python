from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.api.v1.schemas.base import ResponseItem
from app.api.v1.schemas.task.conversion.device import (
    PrecisionPayload,
    SoftwareVersionPayload,
    SupportedDevicePayload,
    TargetDevicePayload,
)
from netspresso.enums.conversion import TARGET_FRAMEWORK_DISPLAY_MAP, Precision, TargetFramework, TargetFrameworkDisplay
from netspresso.enums.device import DeviceName, SoftwareVersion
from netspresso.enums.model import DataType


class TargetFrameworkPayload(BaseModel):
    name: TargetFramework = Field(description="Framework name")
    display_name: Optional[TargetFrameworkDisplay] = Field(default=None, description="Framework display name")

    @model_validator(mode="after")
    def set_display_name(self) -> str:
        self.display_name = TARGET_FRAMEWORK_DISPLAY_MAP.get(self.name)

        return self


class SupportedDeviceResponse(BaseModel):
    framework: TargetFrameworkPayload
    devices: List[SupportedDevicePayload]


class SupportedDevicesResponse(BaseModel):
    data: List[SupportedDeviceResponse]


class ConversionCreate(BaseModel):
    input_model_id: str = Field(description="Input model ID")
    framework: TargetFramework = Field(description="Framework name")
    device_name: DeviceName = Field(description="Device name")
    software_version: Optional[SoftwareVersion] = Field(default=None, description="Software version")
    precision: Precision = Field(description="Precision")
    calibration_dataset_path: Optional[str] = Field(default=None, description="Path to the calibration dataset")


class ConversionPayload(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    task_id: str
    model_id: Optional[str] = None
    framework: TargetFrameworkPayload
    device_name: TargetDevicePayload
    software_version: Optional[SoftwareVersionPayload] = None
    precision: PrecisionPayload
    status: str
    is_deleted: bool
    error_detail: Optional[Dict] = None
    input_model_id: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ConversionResponse(ResponseItem):
    data: ConversionPayload
