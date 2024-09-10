import time
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib import request

from loguru import logger

from netspresso.base import NetsPressoBase
from netspresso.clients.auth import TokenHandler
from netspresso.clients.auth.response_body import UserResponse
from netspresso.clients.launcher import launcher_client_v2
from netspresso.clients.launcher.v2.schemas.task.quantize.request_body import QuantizationOptions
from netspresso.clients.launcher.v2.schemas.task.quantize.response_body import QuantizeTask
from netspresso.enums import (
    QuantizationDataType,
    QuantizationMode,
    ServiceTask,
    SimilarityMetric,
    Status,
    TaskStatusForDisplay,
)
from netspresso.metadata.quantizer import QuantizerMetadata
from netspresso.utils import FileHandler
from netspresso.utils.metadata import MetadataHandler


class Quantizer(NetsPressoBase):
    def __init__(self, token_handler: TokenHandler, user_info: UserResponse):
        """Initialize the Quantizer."""

        super().__init__(token_handler)
        self.user_info = user_info

    def initialize_metadata(
        self,
        output_dir,
        input_model_path,
        threshold,
        quantization_mode,
        metric,
        weight_quantization_bitwidth,
        activation_quantization_bitwidth,
    ):
        def create_metadata_with_status(status, error_message=None):
            metadata = QuantizerMetadata()
            metadata.status = status
            if error_message:
                logger.error(error_message)
            return metadata

        try:
            metadata = QuantizerMetadata()
        except Exception as e:
            error_message = f"An unexpected error occurred during metadata initialization: {e}"
            metadata = create_metadata_with_status(Status.ERROR, error_message)
        except KeyboardInterrupt:
            warning_message = "Quantization task was interrupted by the user."
            metadata = create_metadata_with_status(Status.STOPPED, warning_message)
        finally:
            metadata.input_model_path = Path(input_model_path).resolve().as_posix()
            metadata.quantize_task_info.threshold = threshold
            metadata.quantize_task_info.quantization_mode = quantization_mode
            metadata.quantize_task_info.metric = metric
            metadata.quantize_task_info.weight_quantization_bitwidth = weight_quantization_bitwidth
            metadata.quantize_task_info.activation_quantization_bitwidth = activation_quantization_bitwidth
            MetadataHandler.save_metadata(data=metadata, folder_path=output_dir)

        return metadata

    def _download_quantized_model(self, quantize_task: QuantizeTask, local_path: str) -> None:
        """Download the quantizeed model with given quantization task or quantization task uuid.

        Args:
            quantization_task (QuantizeTask): Launcher Model Object or the uuid of the quantization task.

        Raises:
            e: If an error occurs while getting the quantization task information.
        """

        self.token_handler.validate_token()

        try:
            download_url = launcher_client_v2.quantizer.download_model_file(
                quantize_task_uuid=quantize_task.quantize_task_id,
                access_token=self.token_handler.tokens.access_token,
            ).data.presigned_download_url

            request.urlretrieve(download_url, local_path)
            logger.info(f"Model downloaded at {Path(local_path)}")

        except Exception as e:
            logger.error(f"Download quantized model failed. Error: {e}")
            raise e

    def quantize_model(
        self,
        input_model_path: str,
        output_dir: str,
        dataset_path: Optional[str],
        quantization_mode: QuantizationMode = QuantizationMode.PLAIN_QUANTIZATION,
        metric: SimilarityMetric = SimilarityMetric.SNR,
        threshold: Union[float, int] = 0,
        weight_quantization_bitwidth: QuantizationDataType = QuantizationDataType.INT8,
        activation_quantization_bitwidth: QuantizationDataType = QuantizationDataType.INT8,
        input_layers: List[Dict[str, int]] = None,
        wait_until_done: bool = True,
        sleep_interval: int = 30,
    ) -> QuantizerMetadata:
        """Quantize a model to the specified framework.

        Args:
            input_model_path (str): The file path where the model is located.
            output_dir (str): The local folder path to save the quantized model.
            dataset_path (str): Path to the dataset. Useful for certain quantizations.
            quantization_mode (QuantizationMode): Quantization mode
            metric (SimilarityMetric): Quantization quality metrics.
            threshold (Union[float, int]): Quantization quality threshold
            weight_quantization_bitwidth (QuantizationDataType): Weight quantization bitwidth
            activation_quantization_bitwidth (QuantizationDataType): Activation quantization bitwidth
            input_layers (List[InputShape], optional): Target input shape for quantization (e.g., dynamic batch to static batch).
            wait_until_done (bool): If True, wait for the quantization result before returning the function.
                                If False, request the quantization and return  the function immediately.

        Raises:
            e: If an error occurs during the model quantization.

        Returns:
            QuantizerMetadata: Quantize metadata.
        """

        FileHandler.check_input_model_path(input_model_path)
        output_dir = FileHandler.create_unique_folder(folder_path=output_dir)
        metadata = self.initialize_metadata(
            output_dir=output_dir,
            input_model_path=input_model_path,
            threshold=threshold,
            quantization_mode=quantization_mode,
            metric=metric,
            weight_quantization_bitwidth=weight_quantization_bitwidth,
            activation_quantization_bitwidth=activation_quantization_bitwidth,
        )

        try:
            if metadata.status in [Status.ERROR, Status.STOPPED]:
                return metadata

            self.validate_token_and_check_credit(service_task=ServiceTask.MODEL_QUANTIZE)

            # Get presigned_model_upload_url
            presigned_url_response = launcher_client_v2.quantizer.presigned_model_upload_url(
                access_token=self.token_handler.tokens.access_token,
                input_model_path=input_model_path,
            )

            # Upload model_file
            launcher_client_v2.quantizer.upload_model_file(
                access_token=self.token_handler.tokens.access_token,
                input_model_path=input_model_path,
                presigned_upload_url=presigned_url_response.data.presigned_upload_url,
            )

            # Validate model_file
            validate_model_response = launcher_client_v2.quantizer.validate_model_file(
                access_token=self.token_handler.tokens.access_token,
                input_model_path=input_model_path,
                ai_model_id=presigned_url_response.data.ai_model_id,
            )

            # Start quantize task
            input_layers = input_layers if input_layers else validate_model_response.data.detail.input_layers
            quantization_options = QuantizationOptions(
                metric=metric,
                threshold=threshold,
                weight_quantization_bitwidth=weight_quantization_bitwidth,
                activation_quantization_bitwidth=activation_quantization_bitwidth,
            )
            quantize_response = launcher_client_v2.quantizer.start_task(
                access_token=self.token_handler.tokens.access_token,
                input_model_id=presigned_url_response.data.ai_model_id,
                quantization_mode=quantization_mode,
                quantization_options=quantization_options,
                input_layers=input_layers,
                dataset_path=dataset_path,
            )

            metadata.model_info = validate_model_response.data.to()
            metadata.quantize_task_info = quantize_response.data.to(validate_model_response.data.uploaded_file_name)
            MetadataHandler.save_metadata(data=metadata, folder_path=output_dir)

            if wait_until_done:
                while True:
                    self.token_handler.validate_token()
                    quantize_response = launcher_client_v2.quantizer.read_task(
                        access_token=self.token_handler.tokens.access_token,
                        task_id=quantize_response.data.quantize_task_id,
                    )
                    if quantize_response.data.status in [
                        TaskStatusForDisplay.FINISHED,
                        TaskStatusForDisplay.ERROR,
                        TaskStatusForDisplay.TIMEOUT,
                    ]:
                        break

                    time.sleep(sleep_interval)

            if quantize_response.data.status == TaskStatusForDisplay.FINISHED:
                default_model_path = FileHandler.get_default_model_path(folder_path=output_dir)
                download_model_path = default_model_path.with_suffix(".zip").as_posix()
                self._download_quantized_model(quantize_task=quantize_response.data, local_path=download_model_path)

                metadata.quantized_model_path = download_model_path
                FileHandler.unzip(zip_file_path=download_model_path, target_path=output_dir)
                FileHandler.remove_file(file_path=download_model_path)

                old_file_path = Path(output_dir) / "quantized.onnx"
                quantized_model_path = default_model_path.with_suffix(".onnx").as_posix()
                metadata.quantized_model_path = old_file_path
                FileHandler.rename_file(old_file_path=old_file_path, new_file_path=quantized_model_path)

                compare_result = FileHandler.load_json(file_path=output_dir / "compare_result.json")

                self.print_remaining_credit(service_task=ServiceTask.MODEL_QUANTIZE)

                metadata.status = Status.COMPLETED
                metadata.quantized_model_path = quantized_model_path
                metadata.compare_result = compare_result
                logger.info("Quantization task was completed successfully.")
            else:
                metadata = self.handle_error(metadata, ServiceTask.MODEL_QUANTIZE, quantize_response.data.error_log)

        except Exception as e:
            metadata = self.handle_error(metadata, ServiceTask.MODEL_QUANTIZE, e.args[0])
        except KeyboardInterrupt:
            metadata = self.handle_stop(metadata, ServiceTask.MODEL_QUANTIZE)
        finally:
            MetadataHandler.save_metadata(data=metadata, folder_path=output_dir)

        return metadata

    def get_quantization_task(self, quantization_task_id: str) -> QuantizeTask:
        """Get the quantization task information with given quantization task uuid.

        Args:
            quantization_task_id (str): Quantize task UUID of the quantize task.

        Raises:
            e: If an error occurs during the model quantization.

        Returns:
            QuantizeTask: Model quantization task dictionary.
        """

        self.token_handler.validate_token()

        response = launcher_client_v2.quantizer.read_task(
            access_token=self.token_handler.tokens.access_token,
            task_id=quantization_task_id,
        )
        return response.data

    def cancel_quantization_task(self, quantization_task_id: str) -> QuantizeTask:
        """Cancel the quantization task with given quantization task uuid.

        Args:
            quantization_task_id (str): Quantize task UUID of the quantize task.

        Raises:
            e: If an error occurs during the task cancel.

        Returns:
            QuantizeTask: Model quantization task dictionary.
        """

        self.token_handler.validate_token()

        response = launcher_client_v2.quantizer.cancel_task(
            access_token=self.token_handler.tokens.access_token,
            task_id=quantization_task_id,
        )
        return response.data
