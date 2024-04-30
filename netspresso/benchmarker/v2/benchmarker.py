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
from netspresso.enums import TaskStatusForDisplay, Status, TaskType
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
            benchmark_task_uuid (str): Benchmark task UUID of the benchmark task.

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
    ) -> BenchmarkTask:
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
            Dict: Model benchmark task dictionary.
        """

        FileHandler.check_input_model_path(input_model_path)

        self.token_handler.validate_token()

        file_name = "benchmark"

        try:
            folder_path = Path(input_model_path).parent

            benchmarker_metadata = BenchmarkerMetadata()
            metadatas = []

            if FileHandler.check_exists(folder_path / f"{file_name}.json"):
                metadatas = MetadataHandler.load_json(folder_path / f"{file_name}.json")
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
            model_upload_response = launcher_client_v2.benchmarker.upload_model_file(
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
                input_layer=validate_model_response.data.detail.input_layer,
                software_version=target_software_version,
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
                    ]:
                        break
                    time.sleep(3)

            benchmark_task = response.data
            input_model_info = validate_model_response.data

            benchmarker_metadata.status = Status.COMPLETED
            benchmarker_metadata.task_type = TaskType.BENCHMARK
            benchmarker_metadata.input_model_path = input_model_path
            benchmarker_metadata.benchmark_task_info = benchmark_task.to()
            benchmarker_metadata.convert_task_info = benchmark_task.benchmark_result.to(
                file_size=input_model_info.file_size_in_mb
            )

            metadatas.append(asdict(benchmarker_metadata))
            MetadataHandler.save_json(
                data=metadatas,
                folder_path=folder_path,
                file_name=file_name,
            )
            logger.info(f"BenchmarkMetadata : {benchmarker_metadata}")

            return response.data

        except Exception as e:
            logger.error(f"Benchmark failed. Error: {e}")
            benchmarker_metadata.status = Status.ERROR
            metadatas.append(asdict(benchmarker_metadata))
            MetadataHandler.save_json(
                data=metadatas,
                folder_path=folder_path,
                file_name=file_name,
            )
            raise e

        except KeyboardInterrupt:
            benchmarker_metadata.status = Status.STOPPED
            metadatas.append(asdict(benchmarker_metadata))
            MetadataHandler.save_json(
                data=metadatas,
                folder_path=folder_path,
                file_name=file_name,
            )
