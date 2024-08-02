import time
from pathlib import Path
from typing import Optional, Union

from loguru import logger

from netspresso.clients.auth import TokenHandler, auth_client
from netspresso.clients.auth.response_body import UserResponse
from netspresso.clients.launcher import launcher_client_v2
from netspresso.clients.launcher.v2.schemas.task.benchmark.response_body import (
    BenchmarkTask,
)
from netspresso.enums import Status, TaskStatusForDisplay
from netspresso.enums.credit import ServiceCredit, ServiceTask
from netspresso.enums.device import (
    DeviceName,
    HardwareType,
    SoftwareVersion,
)
from netspresso.metadata.benchmarker import BenchmarkerMetadata
from netspresso.utils import FileHandler
from netspresso.utils.metadata import MetadataHandler


class BenchmarkerV2:
    def __init__(self, token_handler: TokenHandler, user_info: UserResponse) -> None:
        """Initialize the Benchmarker."""

        self.token_handler = token_handler
        self.user_info = user_info

    def check_credit_balance(self, service_task: ServiceTask):
        current_credit = auth_client.get_credit(
            access_token=self.token_handler.tokens.access_token,
            verify_ssl=self.token_handler.verify_ssl
        )
        service_credit = ServiceCredit.get_credit(service_task)
        service_task_name = service_task.name.replace("_", " ").lower()
        if current_credit < service_credit:
            logger.error(
                f"Insufficient balance: {current_credit} credits available, but {service_credit} credits required for {service_task_name} task."
            )
            raise RuntimeError(
                f"Your current balance of {current_credit} credits is insufficient to complete the task. \n{service_credit} credits are required for one {service_task_name} task. \nFor additional credit, please contact us at netspresso@nota.ai."
            )

    def print_remaining_credit(self, service_task):
        if launcher_client_v2.is_cloud():
            service_credit = ServiceCredit.get_credit(service_task)
            remaining_credit = auth_client.get_credit(
                self.token_handler.tokens.access_token, verify_ssl=self.token_handler.verify_ssl
            )
            logger.info(
                f"{service_credit} credits have been consumed. Remaining Credit: {remaining_credit}"
            )

    def initialize_metadata(self, input_model_path: str):
        def create_metadata_with_status(status, error_message=None):
            metadata = BenchmarkerMetadata()
            metadata.status = status
            if error_message:
                logger.error(error_message)
            return metadata

        try:
            metadata = BenchmarkerMetadata()
        except Exception as e:
            error_message = f"An unexpected error occurred during metadata initialization: {e}"
            metadata = create_metadata_with_status(Status.ERROR, error_message)
        except KeyboardInterrupt:
            warning_message = "Benchmark task was interrupted by the user."
            metadata = create_metadata_with_status(Status.STOPPED, warning_message)
        finally:
            # Load existing metadata if available
            metadatas = []
            output_dir = Path(input_model_path).parent
            file_path = output_dir / "benchmark.json"
            if FileHandler.check_exists(file_path):
                metadatas = MetadataHandler.load_json(file_path)

            # Add current metadata and save
            metadatas.append(metadata)
            MetadataHandler.save_benchmark_result(data=metadatas, folder_path=output_dir)

        return metadatas

    def handle_error(self, metadata, error_message):
        metadata.status = Status.ERROR
        metadata.update_message(exception_detail=error_message)
        logger.error(f"Benchmark task failed due to an error: {error_message}")

        return metadata

    def benchmark_model(
        self,
        input_model_path: str,
        target_device_name: DeviceName,
        target_software_version: Optional[Union[str, SoftwareVersion]] = None,
        target_hardware_type: Optional[Union[str, HardwareType]] = None,
        wait_until_done: bool = True,
        sleep_interval: int = 30,
    ) -> BenchmarkerMetadata:
        """Benchmark the specified model on the specified device.

        Args:
            input_model_path (str): The file path where the model is located.
            target_device_name (DeviceName): Target device name.
            target_software_version (Union[str, SoftwareVersion], optional): Target software version. Required if target_device_name is one of the Jetson devices.
            target_hardware_type (Union[str, HardwareType], optional): Hardware type. Acceleration options for processing the model inference.
            wait_until_done (bool): If True, wait for the conversion result before returning the function.
                                If False, request the conversion and return the function immediately.

        Raises:
            e: If an error occurs during the benchmarking of the model.

        Returns:
            BenchmarkerMetadata: Benchmark metadata.
        """

        FileHandler.check_input_model_path(input_model_path)
        metadatas = self.initialize_metadata(input_model_path=input_model_path)

        try:
            metadata: BenchmarkerMetadata = metadatas[-1]
            output_dir = Path(input_model_path).parent
            metadata.input_model_path = Path(input_model_path).resolve().as_posix()
            if metadata.status in [Status.ERROR, Status.STOPPED]:
                return metadata

            self.token_handler.validate_token()
            self.check_credit_balance(service_task=ServiceTask.MODEL_CONVERT)

            # Get presigned_model_upload_url
            presigned_url_response = launcher_client_v2.benchmarker.presigned_model_upload_url(
                access_token=self.token_handler.tokens.access_token,
                input_model_path=input_model_path,
            )

            # Upload model_file
            launcher_client_v2.benchmarker.upload_model_file(
                access_token=self.token_handler.tokens.access_token,
                input_model_path=input_model_path,
                presigned_upload_url=presigned_url_response.data.presigned_upload_url,
            )

            # Validate model_file
            validate_model_response = launcher_client_v2.benchmarker.validate_model_file(
                access_token=self.token_handler.tokens.access_token,
                input_model_path=input_model_path,
                ai_model_id=presigned_url_response.data.ai_model_id,
            )

            # Start benchmark task
            benchmark_response = launcher_client_v2.benchmarker.start_task(
                access_token=self.token_handler.tokens.access_token,
                input_model_id=presigned_url_response.data.ai_model_id,
                data_type=validate_model_response.data.detail.data_type,
                target_device_name=target_device_name,
                hardware_type=target_hardware_type,
                input_layer=validate_model_response.data.detail.input_layers[0],
                software_version=target_software_version,
            )

            metadata.benchmark_task_info = benchmark_response.data.to()
            MetadataHandler.save_benchmark_result(data=metadatas, folder_path=output_dir)

            if wait_until_done:
                while True:
                    self.token_handler.validate_token()
                    benchmark_response = launcher_client_v2.benchmarker.read_task(
                        access_token=self.token_handler.tokens.access_token,
                        task_id=benchmark_response.data.benchmark_task_id,
                    )
                    if benchmark_response.data.status in [
                        TaskStatusForDisplay.FINISHED,
                        TaskStatusForDisplay.ERROR,
                        TaskStatusForDisplay.TIMEOUT,
                    ]:
                        break

                    time.sleep(sleep_interval)

            if benchmark_response.data.status == TaskStatusForDisplay.FINISHED:
                self.print_remaining_credit(service_task=ServiceTask.MODEL_BENCHMARK)
                metadata.status = Status.COMPLETED
                metadata.benchmark_result = benchmark_response.data.benchmark_result.to(
                    file_size=validate_model_response.data.file_size_in_mb
                )
                logger.info("Benchmark task was completed successfully.")
            else:
                metadata = self.handle_error(metadata, benchmark_response.data.error_log)

        except Exception as e:
            metadata = self.handle_error(metadata, e.args[0])
        except KeyboardInterrupt:
            metadata.status = Status.STOPPED
            logger.error("Benchmark task was interrupted by the user.")
        finally:
            metadatas[-1] = metadata
            MetadataHandler.save_benchmark_result(data=metadatas, folder_path=output_dir)

        return metadata

    def get_benchmark_task(self, benchmark_task_id: str) -> BenchmarkTask:
        """Get information about the specified benchmark task using the benchmark task UUID.

        Args:
            benchmark_task_id (str): Benchmark task UUID of the benchmark task.

        Raises:
            e: If an error occurs while retrieving information about the benchmark task.

        Returns:
            BenchmarkTask: Model benchmark task object.
        """

        self.token_handler.validate_token()

        response = launcher_client_v2.benchmarker.read_task(
            access_token=self.token_handler.tokens.access_token,
            task_id=benchmark_task_id,
        )
        return response.data
