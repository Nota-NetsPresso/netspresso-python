import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Union

from loguru import logger

from netspresso.clients.auth import TokenHandler, auth_client
from netspresso.clients.auth.response_body import UserResponse
from netspresso.clients.launcher import launcher_client_v2
from netspresso.clients.launcher.v2.schemas.task.benchmark.response_body import (
    BenchmarkTask,
)
from netspresso.enums import Status, TaskStatusForDisplay, TaskType
from netspresso.enums.credit import ServiceCredit
from netspresso.enums.device import (
    DeviceName,
    HardwareType,
    SoftwareVersion,
)
from netspresso.metadata.benchmarker import BenchmarkerMetadata
from netspresso.utils import FileHandler, check_credit_balance
from netspresso.utils.metadata import MetadataHandler


class BenchmarkerV2:
    def __init__(self, token_handler: TokenHandler, user_info: UserResponse) -> None:
        """Initialize the Benchmarker."""

        self.token_handler = token_handler
        self.user_info = user_info

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

    def benchmark_model(
        self,
        input_model_path: str,
        target_device_name: DeviceName,
        target_software_version: Optional[Union[str, SoftwareVersion]] = None,
        target_hardware_type: Optional[Union[str, HardwareType]] = None,
        wait_until_done: bool = True,
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

        self.token_handler.validate_token()

        file_name = "benchmark"

        try:
            folder_path = Path(input_model_path).parent

            benchmarker_metadata = BenchmarkerMetadata()
            benchmarker_metadata.input_model_path = input_model_path
            metadatas = []

            file_path = folder_path / f"{file_name}.json"

            if FileHandler.check_exists(file_path):
                metadatas = MetadataHandler.load_json(file_path)
                # metadatas.append(asdict(benchmarker_metadata))

            current_credit = auth_client.get_credit(
                access_token=self.token_handler.tokens.access_token,
                verify_ssl=self.token_handler.verify_ssl,
            )
            check_credit_balance(
                user_credit=current_credit, service_credit=ServiceCredit.MODEL_BENCHMARK
            )

            # GET presigned_model_upload_url
            presigned_url_response = (
                launcher_client_v2.benchmarker.presigned_model_upload_url(
                    access_token=self.token_handler.tokens.access_token,
                    input_model_path=input_model_path,
                )
            )

            # UPLOAD model_file
            launcher_client_v2.benchmarker.upload_model_file(
                access_token=self.token_handler.tokens.access_token,
                input_model_path=input_model_path,
                presigned_upload_url=presigned_url_response.data.presigned_upload_url,
            )

            # VALIDATE model_file
            validate_model_response = (
                launcher_client_v2.benchmarker.validate_model_file(
                    access_token=self.token_handler.tokens.access_token,
                    input_model_path=input_model_path,
                    ai_model_id=presigned_url_response.data.ai_model_id,
                )
            )

            # START convert task
            response = launcher_client_v2.benchmarker.start_task(
                access_token=self.token_handler.tokens.access_token,
                input_model_id=presigned_url_response.data.ai_model_id,
                data_type=validate_model_response.data.detail.data_type,
                target_device_name=target_device_name,
                hardware_type=target_hardware_type,
                input_layer=validate_model_response.data.detail.input_layers[0],
                software_version=target_software_version,
            )

            benchmarker_metadata.benchmark_task_info = response.data.to()
            metadatas.append(asdict(benchmarker_metadata))
            MetadataHandler.save_json(
                data=metadatas,
                folder_path=folder_path,
                file_name=file_name,
            )

            if wait_until_done:
                while True:
                    # Poll Benchmark Task status
                    response = launcher_client_v2.benchmarker.read_task(
                        access_token=self.token_handler.tokens.access_token,
                        task_id=response.data.benchmark_task_id,
                    )
                    if response.data.status in [
                        TaskStatusForDisplay.FINISHED,
                        TaskStatusForDisplay.ERROR,
                        TaskStatusForDisplay.TIMEOUT,
                    ]:
                        break
                    self.token_handler.validate_token()
                    time.sleep(3)

            if launcher_client_v2.is_cloud():
                remaining_credit = auth_client.get_credit(
                    self.token_handler.tokens.access_token,
                    self.token_handler.verify_ssl,
                )
                logger.info(
                    f"{ServiceCredit.MODEL_BENCHMARK} credits have been consumed. Remaining Credit: {remaining_credit}"
                )

            benchmark_task = response.data
            input_model_info = validate_model_response.data

            if benchmark_task.status == TaskStatusForDisplay.FINISHED:
                benchmarker_metadata.status = Status.COMPLETED
                logger.info("Benchmark task successfully completed.")
            else:
                benchmarker_metadata.status = Status.ERROR
                logger.info("Benchmark task failed with an error.")

            benchmarker_metadata.benchmark_result = benchmark_task.benchmark_result.to(
                file_size=input_model_info.file_size_in_mb
            )

            metadatas[-1] = asdict(benchmarker_metadata)
            MetadataHandler.save_json(
                data=metadatas,
                folder_path=folder_path,
                file_name=file_name,
            )

            return benchmarker_metadata

        except Exception as e:
            logger.error(f"Benchmark failed. Error: {e}")
            benchmarker_metadata.status = Status.ERROR
            metadatas[-1] = asdict(benchmarker_metadata)
            MetadataHandler.save_json(
                data=metadatas,
                folder_path=folder_path,
                file_name=file_name,
            )
            raise e

        except KeyboardInterrupt:
            benchmarker_metadata.status = Status.STOPPED
            metadatas[-1] = asdict(benchmarker_metadata)
            MetadataHandler.save_json(
                data=metadatas,
                folder_path=folder_path,
                file_name=file_name,
            )
