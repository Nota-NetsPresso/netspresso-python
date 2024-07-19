from pathlib import Path
from typing import List, Optional, Tuple, Union

from loguru import logger
import qai_hub as hub
from qai_hub.client import CompileJob, Dataset, Device
from qai_hub.public_rest_api import DatasetEntries

from netspresso.enums import Status
from netspresso.qai_hub.base import QAIHubBase
from netspresso.qai_hub.options import Runtime, CompileOptions, Framework, Extension

from netspresso.metadata.converter import ConverterMetadata
from netspresso.utils import FileHandler
from netspresso.utils.metadata import MetadataHandler


class QAIHubConverter(QAIHubBase):
    def get_target_extension(self, runtime=Runtime.TFLITE):
        runtime_extensions = {
            Runtime.TFLITE: ".tflite",
            Runtime.QNN_LIB_AARCH64_ANDROID: ".so",
            Runtime.QNN_CONTEXT_BINARY: ".bin",
            Runtime.ONNX: ".onnx",
            Runtime.PRECOMPILED_QNN_ONNX: ".zip"
        }

        return runtime_extensions.get(runtime, None)

    def get_framework(self, extension: Extension):
        if extension == Extension.ONNX:
            return Framework.ONNX
        elif extension == Extension.PT:
            return Framework.PYTORCH
        elif extension == Extension.AIMET:
            return Framework.AIMET
        elif extension == Extension.H5:
            return Framework.TENSORFLOW

    def get_source_extension(self, model_path):
        extension = Path(model_path).suffix

        return extension

    def get_display_runtime(self, runtime: Runtime) -> str:
        RUNTIME_DISPLAY_MAP = {
            Runtime.TFLITE: "TensorFlow Lite",
            Runtime.QNN_LIB_AARCH64_ANDROID: "Qualcomm® AI Engine Direct model library targeting AArch64 Android",
            Runtime.QNN_CONTEXT_BINARY: "Qualcomm® AI Engine Direct context binary targeting the hardware specified in the compile job.",
            Runtime.ONNX: "ONNX",
            Runtime.PRECOMPILED_QNN_ONNX: "ONNX Runtime model with a pre-compiled QNN context binary.",

        }
        return RUNTIME_DISPLAY_MAP.get(runtime, "Unknown runtime")

    def transform_shape(self, data):
        for value in data.values():
            shape, _ = value
            return {
                "batch": shape[0],
                "channel": shape[1],
                "dimension": list(shape[2:])
            }

    def dict_to_tuple(self, data):
        batch = data.get("batch")
        channel = data.get("channel")
        dimension = data.get("dimension", [])

        if batch is not None and channel is not None and isinstance(dimension, list):
            return (batch, channel, *dimension)
        else:
            raise ValueError("Invalid input data format")

    def convert_model(
        self,
        input_model_path: Union[str, Path],
        output_dir: str,
        target_device_name: Union[Device, List[Device]],
        input_shape: Union[List, Tuple, None] = None,
        options: CompileOptions = CompileOptions(),
        job_name: Optional[str] = None,
        single_compile: bool = True,
        calibration_data: Union[Dataset, DatasetEntries, str, None] = None,
        retry: bool = True,
        wait_until_done: bool = True,
    ) -> Union[ConverterMetadata, List[ConverterMetadata]]:

        output_dir = FileHandler.create_unique_folder(folder_path=output_dir)
        default_model_path = (Path(output_dir) / f"{Path(output_dir).name}.ext").resolve()
        metadata = ConverterMetadata()
        metadata.input_model_path = Path(input_model_path).resolve().as_posix()
        extension = self.get_source_extension(model_path=input_model_path)
        metadata.model_info.framework = self.get_framework(extension=extension)

        MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_dir)

        target_extension = self.get_target_extension(runtime=options.target_runtime)
        converted_model_path = default_model_path.with_suffix(target_extension).as_posix()
        cli_string = options.to_cli_string()

        if isinstance(input_shape, List):
            input_shape = tuple(input_shape)

        job = hub.submit_compile_job(
            model=input_model_path,
            device=target_device_name,
            name=job_name,
            input_specs={'image': input_shape},
            options=cli_string,
            single_compile=single_compile,
            calibration_data=calibration_data,
            retry=retry,
        )

        # metadata.model_info.input_shapes = [input_shape]
        metadata.model_info.input_shapes = [self.transform_shape(job.shapes)]
        metadata.model_info.data_type = job.shapes["image"][1]
        metadata.convert_task_info.convert_task_uuid = job.job_id
        metadata.convert_task_info.input_model_uuid = job.model.model_id
        metadata.convert_task_info.device_name = target_device_name.name
        metadata.convert_task_info.display_device_name = target_device_name.name
        metadata.convert_task_info.framework = options.target_runtime
        metadata.convert_task_info.display_framework = self.get_display_runtime(options.target_runtime)

        MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_dir)

        if wait_until_done:
            job: CompileJob = hub.get_job(job.job_id)
            status = job.wait()

            if status.success:
                logger.info(f"{status.symbol} {status.state.name}")
                self.download_model(job=job, filename=converted_model_path)
                target_model = job.get_target_model()
                metadata.convert_task_info.output_model_uuid = target_model.model_id
                metadata.converted_model_path = converted_model_path
                metadata.convert_task_info.data_type = job.target_shapes["image"][1]
                metadata.available_options = job.compatible_devices
                metadata.status = Status.COMPLETED
            else:
                logger.info(f"{status.symbol} {status.state}: {status.message}")
                metadata.status = Status.ERROR
                metadata.update_message(exception_detail=status.message)

        MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_dir)
        
        return metadata

    def download_model(self, job: CompileJob, filename: str):
        job.download_target_model(filename=filename)
