from enum import Enum


class SourceFramework(str, Enum):
    ONNX = "onnx"


class SourceFrameworkDisplay(str, Enum):
    ONNX = "ONNX"


SOURCE_FRAMEWORK_DISPLAY_MAP = {
    SourceFramework.ONNX: SourceFrameworkDisplay.ONNX,
}


class TargetFramework(str, Enum):
    TENSORRT = "tensorrt"
    TENSORFLOW_LITE = "tensorflow_lite"
    OPENVINO = "openvino"
    DRPAI = "drpai"


class TargetFrameworkDisplay(str, Enum):
    TENSORRT = "TensorRT"
    TENSORFLOW_LITE = "TensorFlow Lite"
    OPENVINO = "OpenVINO"
    DRPAI = "DRPAI"


TARGET_FRAMEWORK_DISPLAY_MAP = {
    TargetFramework.TENSORRT: TargetFrameworkDisplay.TENSORRT,
    TargetFramework.TENSORFLOW_LITE: TargetFrameworkDisplay.TENSORFLOW_LITE,
    TargetFramework.OPENVINO: TargetFrameworkDisplay.OPENVINO,
    TargetFramework.DRPAI: TargetFrameworkDisplay.DRPAI,
}


class Precision(str, Enum):
    FP16 = "FP16"
    INT8 = "INT8"


class PrecisionDisplay(str, Enum):
    FP16 = "FP16"
    INT8 = "INT8"


PRECISION_DISPLAY_MAP = {
    Precision.FP16: PrecisionDisplay.FP16,
    Precision.INT8: PrecisionDisplay.INT8,
}
