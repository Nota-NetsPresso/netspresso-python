from enum import Enum


class Framework(str, Enum):
    TENSORFLOW_KERAS = "tensorflow_keras"
    TENSORFLOW = "saved_model"
    ONNX = "onnx"


class FrameworkDisplay(str, Enum):
    TENSORFLOW_KERAS = "TensorFlow Keras"
    TENSORFLOW = "TensorFlow"
    ONNX = "ONNX"


FRAMEWORK_DISPLAY_MAP = {
    Framework.TENSORFLOW_KERAS: FrameworkDisplay.TENSORFLOW_KERAS,
    Framework.TENSORFLOW: FrameworkDisplay.TENSORFLOW,
    Framework.ONNX: FrameworkDisplay.ONNX,
}
