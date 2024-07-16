import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

from loguru import logger
import qai_hub as hub
from qai_hub.client import CompileJob, Dataset, Device, Model, SourceModel
from qai_hub.public_rest_api import DatasetEntries

from netspresso.qai_hub.base import QAIHubBase
from netspresso.qai_hub.options.common import Runtime, CompileOptions
from netspresso.utils import FileHandler


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

    def convert_model(
        self,
        model: Union[Model, SourceModel, str, Path],
        output_dir: str,
        device: Union[Device, List[Device]],
        name: Optional[str] = None,
        input_shape: Union[List, Tuple, None] = None,
        options: CompileOptions = CompileOptions(),
        single_compile: bool = True,
        calibration_data: Union[Dataset, DatasetEntries, str, None] = None,
        retry: bool = True,
        wait_until_done: bool = True,
    ) -> Union[CompileJob, List[CompileJob]]:

        output_dir = FileHandler.create_unique_folder(folder_path=output_dir)
        target_extension = self.get_target_extension()
        cli_string = options.to_cli_string()

        if isinstance(input_shape, List):
            input_shape = tuple(input_shape)

        compile_job = hub.submit_compile_job(
            model=model,
            device=device,
            name=name,
            input_specs={'image': input_shape},
            options=cli_string,
            single_compile=single_compile,
            calibration_data=calibration_data,
            retry=retry,
        )

        if wait_until_done:
            compile_job = hub.get_job(compile_job.job_id)
            status = compile_job.wait()

            if status.success:
                logger.info(f"{status.symbol} {status.state}")
                self.download_model(job=compile_job, filename=f"{output_dir}/{name}{target_extension}")
            else:
                logger.info(f"{status.symbol} {status.state}: {status.message}")

        return compile_job

    def download_model(self, job: CompileJob, filename: str):
        job.download_target_model(filename=filename)
