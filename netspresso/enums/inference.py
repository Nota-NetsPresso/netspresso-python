from enum import Enum

from netspresso.exceptions.inferencer import NotSupportedSuffixException


class Runtime(str, Enum):
    ONNX = "onnx"
    TFLITE = "tflite"

    @classmethod
    def get_runtime_by_suffix(cls, suffix: str) -> str:
        suffix_map = {
            ".onnx": cls.ONNX,
            ".tflite": cls.TFLITE,
        }
        runtime = suffix_map.get(suffix.lower(), None)

        if runtime is None:
            raise NotSupportedSuffixException(available_suffixes=list(suffix_map.keys()), suffix=suffix)

        return runtime
