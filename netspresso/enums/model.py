from enum import Enum
from typing import Literal


class Framework(str, Enum):
    TENSORFLOW_KERAS = "tensorflow_keras"
    TENSORFLOW = "saved_model"
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"
    TENSORFLOW_LITE = "tensorflow_lite"
    DRPAI = "drpai"
    DLC = "dlc"

    @classmethod
    def create_compressor_literal(cls):
        return Literal["tensorflow_keras", "pytorch", "onnx"]

    @classmethod
    def create_launcher_literal(cls):
        return Literal[
            "onnx",
            "tensorrt",
            "openvino",
            "tensorflow_lite",
            "drpai",
            "keras",
            "saved_model",
            "dlc"
        ]


class Extension(str, Enum):
    H5 = "h5"
    ZIP = "zip"
    PT = "pt"
    ONNX = "onnx"

    @classmethod
    def create_literal(cls):
        return Literal["h5", "zip", "pt", "onnx"]


class OriginFrom(str, Enum):
    CUSTOM = "custom"
    NPMS = "npms"

    @classmethod
    def create_literal(cls):
        return Literal["custom", "npms"]


class DataType(str, Enum):
    FP32 = "FP32"
    FP16 = "FP16"
    INT8 = "INT8"
    NONE = ""

    @classmethod
    def create_literal(cls):
        return Literal["FP32", "FP16", "INT8", ""]


compressor_framework_literal = Framework.create_compressor_literal()
launcher_framework_literal = Framework.create_launcher_literal()
extension_literal = Extension.create_literal()
originfrom_literal = OriginFrom.create_literal()
datatype_literal = DataType.create_literal()
