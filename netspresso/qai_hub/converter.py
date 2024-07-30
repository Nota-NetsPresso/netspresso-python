from pathlib import Path
from typing import List, Optional, Union

import qai_hub as hub
from loguru import logger
from qai_hub.client import CompileJob, Dataset, Device, InputSpecs
from qai_hub.public_rest_api import DatasetEntries

from netspresso.enums import Status
from netspresso.metadata.converter import ConverterMetadata
from netspresso.qai_hub.base import QAIHubBase
from netspresso.qai_hub.options import CompileOptions
from netspresso.utils import FileHandler
from netspresso.utils.metadata import MetadataHandler


class QAIHubConverter(QAIHubBase):
    def transform_shape(self, data):
        for value in data.values():
            shape, _ = value
            return {"batch": shape[0], "channel": shape[1], "dimension": list(shape[2:])}

    def convert_image_dict_to_list(self, image_dict):
        result = []
        for key, value in image_dict.items():
            batch, channel, *dimension = value
            result.append({
                "name": key,
                "batch": batch,
                "channel": channel,
                "dimension": dimension
            })
        return result

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
        input_shapes: Optional[InputSpecs] = None,
        options: Union[CompileOptions, str] = CompileOptions(),
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

        try:
            target_extension = self.get_target_extension(runtime=options.target_runtime)
            converted_model_path = default_model_path.with_suffix(target_extension).as_posix()

            cli_string = options.to_cli_string() if isinstance(options, CompileOptions) else options

            job = hub.submit_compile_job(
                model=input_model_path,
                device=target_device_name,
                name=job_name,
                input_specs=input_shapes,
                options=cli_string,
                single_compile=single_compile,
                calibration_data=calibration_data,
                retry=retry,
            )

            framework = self.get_framework_by_runtime(options.target_runtime)
            display_framework = self.get_display_framework(framework)

            # metadata.model_info.input_shapes = [input_shape]
            metadata.model_info.input_shapes = self.convert_image_dict_to_list(input_shapes)
            metadata.model_info.data_type = job.shapes["image"][1]
            metadata.convert_task_info.convert_task_uuid = job.job_id
            metadata.convert_task_info.input_model_uuid = job.model.model_id
            metadata.convert_task_info.device_name = target_device_name.name
            metadata.convert_task_info.display_device_name = target_device_name.name
            metadata.convert_task_info.framework = framework
            metadata.convert_task_info.display_framework = display_framework

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

        except KeyboardInterrupt:
            metadata.status = Status.STOPPED
            MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_dir)

    def download_model(self, job: CompileJob, filename: str):
        job.download_target_model(filename=filename)
