from pathlib import Path
from typing import List, Optional, Union

import qai_hub as hub
from qai_hub.client import Device, Job, SourceModelType

from netspresso.qai_hub.options import Extension, Framework, Runtime


class QAIHubBase:
    def get_devices(self, name: str = "", os: str = "", attributes: Union[str, List[str]] = None) -> List[Device]:
        if attributes is None:
            attributes = []
        devices = hub.get_devices(name=name, os=os, attributes=attributes)

        return devices

    def get_device_attributes(self) -> List[str]:
        device_attributes = hub.get_device_attributes()

        return device_attributes

    def get_jobs(self, offset: int = 0, limit: int = 50, creator: Optional[str] = None) -> List[Job]:
        jobs = hub.get_jobs(offset=offset, limit=limit, creator=creator)

        return jobs

    def get_job(self, job_id: str) -> Job:
        job = hub.get_job(job_id=job_id)

        return job

    def get_source_extension(self, model_path):
        extension = Path(model_path).suffix

        return extension

    def get_framework(self, extension: Extension):
        if extension == Extension.ONNX:
            return Framework.ONNX
        elif extension == Extension.PT:
            return Framework.PYTORCH
        elif extension == Extension.AIMET:
            return Framework.AIMET
        elif extension == Extension.H5:
            return Framework.TENSORFLOW

    def get_target_extension(self, runtime=Runtime.TFLITE):
        runtime_extensions = {
            Runtime.TFLITE: ".tflite",
            Runtime.QNN_LIB_AARCH64_ANDROID: ".so",
            Runtime.QNN_CONTEXT_BINARY: ".bin",
            Runtime.ONNX: ".onnx",
            Runtime.PRECOMPILED_QNN_ONNX: ".zip",
        }

        return runtime_extensions.get(runtime)

    def get_display_runtime(self, runtime: Runtime) -> str:
        RUNTIME_DISPLAY_MAP = {
            Runtime.TFLITE: "TensorFlow Lite",
            Runtime.QNN_LIB_AARCH64_ANDROID: "Qualcomm® AI Engine Direct model library targeting AArch64 Android",
            Runtime.QNN_CONTEXT_BINARY: "Qualcomm® AI Engine Direct context binary targeting the hardware specified in the compile job.",
            Runtime.ONNX: "ONNX",
            Runtime.PRECOMPILED_QNN_ONNX: "ONNX Runtime model with a pre-compiled QNN context binary.",
        }
        return RUNTIME_DISPLAY_MAP.get(runtime, "Unknown runtime")

    def get_framework_by_runtime(self, runtime: Runtime):
        FRAMEWORK_RUNTIME_MAP = {
            Runtime.TFLITE: Framework.TFLITE,
            Runtime.QNN_LIB_AARCH64_ANDROID: Framework.QNN,
            Runtime.QNN_CONTEXT_BINARY: Framework.QNN,
            Runtime.ONNX: Framework.ONNX,
            Runtime.PRECOMPILED_QNN_ONNX: Framework.QNN,
        }
        return FRAMEWORK_RUNTIME_MAP.get(runtime, "Unknown framework")

    def get_framework_by_model_type(self, model_type: SourceModelType):
        FRAMEWORK_MODEL_TYPE_MAP = {
            SourceModelType.TORCHSCRIPT: Framework.PYTORCH,
            SourceModelType.TFLITE: Framework.TFLITE,
            SourceModelType.ONNX: Framework.ONNX,
            SourceModelType.ORT: Framework.ONNXRUNTIME,
            SourceModelType.MLMODEL: Framework.COREML,
            SourceModelType.MLMODELC: Framework.COREML,
            SourceModelType.MLPACKAGE: Framework.COREML,
            SourceModelType.TETRART: Framework.TENSORRT,
            SourceModelType.QNN_LIB_AARCH64_ANDROID: Framework.QNN,
            SourceModelType.QNN_LIB_X86_64_LINUX: Framework.QNN,
            SourceModelType.QNN_CONTEXT_BINARY: Framework.QNN,
            SourceModelType.AIMET_ONNX: Framework.AIMET,
            SourceModelType.AIMET_PT: Framework.AIMET,
        }
        return FRAMEWORK_MODEL_TYPE_MAP.get(model_type, "Unknown framework")

    def get_display_framework(self, framework: Framework):
        RUNTIME_DISPLAY_MAP = {
            Framework.PYTORCH: "PyTorch",
            Framework.ONNX: "ONNX",
            Framework.ONNXRUNTIME: "ONNXRuntime",
            Framework.AIMET: "AIMET",
            Framework.TENSORFLOW: "TensorFlow",
            Framework.TFLITE: "Tensorflow Lite",
            Framework.COREML: "CoreML",
            Framework.TENSORRT: "TensorRT",
            Framework.QNN: "QNN",
        }
        return RUNTIME_DISPLAY_MAP.get(framework, "Unknown runtime")
