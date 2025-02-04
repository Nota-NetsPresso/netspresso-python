from pathlib import Path
from typing import List, Optional, Union

import qai_hub as hub
from loguru import logger
from qai_hub import QuantizeDtype
from qai_hub.client import Dataset, QuantizeJob
from qai_hub.public_rest_api import DatasetEntries

from netspresso.enums import Status
from netspresso.metadata.quantizer import QuantizerMetadata
from netspresso.np_qai.base import NPQAIBase
from netspresso.np_qai.options.quantize import QuantizeOptions
from netspresso.utils import FileHandler
from netspresso.utils.metadata import MetadataHandler


class NPQAIQuantizer(NPQAIBase):
    def get_quantize_task_status(self, quantize_task_uuid):
        job: QuantizeJob = hub.get_job(quantize_task_uuid)
        status = job.get_status()

        return status

    def update_quantize_task(self, metadata: QuantizerMetadata):
        job: QuantizeJob = hub.get_job(metadata.quantize_info.quantize_task_uuid)
        status = job.wait()

        if status.success:
            logger.info(f"{status.symbol} {status.state.name}")
            self.download_model(job=job, filename=metadata.quantized_model_path                                         )
            target_model = job.get_target_model()
            metadata.quantize_info.output_model_uuid = target_model.model_id
            metadata.status = Status.COMPLETED
        elif status.failure:
            logger.info(f"{status.symbol} {status.state}: {status.message}")
            metadata.status = Status.ERROR
            metadata.update_message(exception_detail=status.message)

        MetadataHandler.save_json(data=metadata.asdict(), folder_path=Path(metadata.quantized_model_path).parent.as_posix())

        return metadata

    def quantize_model(
        self,
        input_model_path: Union[str, Path],
        output_dir: str,
        weights_dtype: QuantizeDtype,
        activations_dtype: QuantizeDtype,
        options: Union[QuantizeOptions, str] = QuantizeOptions(),
        job_name: Optional[str] = None,
        calibration_data: Union[Dataset, DatasetEntries, str, None] = None,
    ) -> Union[QuantizerMetadata, List[QuantizerMetadata]]:

        output_dir = FileHandler.create_unique_folder(folder_path=output_dir)
        default_model_path = (Path(output_dir) / f"{Path(output_dir).name}.ext").resolve()
        metadata = QuantizerMetadata()
        metadata.input_model_path = Path(input_model_path).resolve().as_posix()
        extension = self.get_source_extension(model_path=input_model_path)
        metadata.model_info.framework = self.get_framework(extension=extension)

        MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_dir)

        try:
            # import ipdb; ipdb.set_trace()
            quantized_model_path = default_model_path.with_suffix(".onnx").as_posix()

            cli_string = options.to_cli_string() if isinstance(options, QuantizeOptions) else options

            job = hub.submit_quantize_job(
                model=input_model_path,
                calibration_data=calibration_data,
                weights_dtype=weights_dtype,
                activations_dtype=activations_dtype,
                name=job_name,
                options=cli_string,
            )
            metadata.quantized_model_path = quantized_model_path
            metadata.quantize_info.quantize_task_uuid = job.job_id
            metadata.quantize_info.input_model_uuid = job.model.model_id
            # metadata.quantize_info.weight_precision = weights_dtype
            # metadata.quantize_info.activation_precision = activations_dtype

            MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_dir)

        except KeyboardInterrupt:
            metadata.status = Status.STOPPED
            MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_dir)

        return metadata

    def download_model(self, job: QuantizeJob, filename: str):
        job.download_target_model(filename=filename)
