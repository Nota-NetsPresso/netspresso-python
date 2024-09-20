from typing import List

from netspresso.exceptions.common import AdditionalData, PyNPException


class NotSupportedDeviceException(PyNPException):
    def __init__(self, available_devices: List, device: int):
        message = f"The device supports {available_devices}. The entered device is {device}."
        super().__init__(
            data=AdditionalData(origin="pynp"),
            error_code="",
            name=self.__class__.__name__,
            message=message,
        )


class NotSupportedDataTypeException(PyNPException):
    def __init__(self, available_data_types: List, data_type: int):
        message = f"The data type supports {available_data_types}. The entered data type is {data_type}."
        super().__init__(
            data=AdditionalData(origin="pynp"),
            error_code="",
            name=self.__class__.__name__,
            message=message,
        )


class NotSupportedSoftwareVersionException(PyNPException):
    def __init__(self, available_software_versions: List, software_version: int):
        message = f"The software_version supports {available_software_versions}. The entered software version is {software_version}."
        super().__init__(
            data=AdditionalData(origin="pynp"),
            error_code="",
            name=self.__class__.__name__,
            message=message,
        )
