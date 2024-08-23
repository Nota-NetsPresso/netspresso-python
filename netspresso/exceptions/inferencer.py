from typing import List

from netspresso.exceptions.common import AdditionalData, PyNPException


class NotSupportedSuffixException(PyNPException):
    def __init__(self, available_suffixes: List, suffix: str):
        message = f"The suffix supports {available_suffixes}. The entered suffix is {suffix}."
        super().__init__(
            data=AdditionalData(origin="pynp"),
            error_code="",
            name=self.__class__.__name__,
            message=message,
        )
