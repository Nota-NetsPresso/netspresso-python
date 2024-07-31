import time
from pathlib import Path
from typing import Optional, Union
from urllib import request

from loguru import logger

from netspresso.clients.auth import TokenHandler, auth_client
from netspresso.clients.auth.response_body import UserResponse
from netspresso.clients.launcher import launcher_client_v2
from netspresso.clients.launcher.v2.schemas import InputLayer
from netspresso.clients.launcher.v2.schemas.task.convert.response_body import ConvertTask
from netspresso.enums import (
    DataType,
    DeviceName,
    Framework,
    ServiceCredit,
    ServiceTask,
    SoftwareVersion,
    Status,
    TaskStatusForDisplay,
)
from netspresso.metadata.converter import ConverterMetadata
from netspresso.utils import FileHandler
from netspresso.utils.metadata import MetadataHandler


class ConverterV2:
    def __init__(self, token_handler: TokenHandler, user_info: UserResponse):
        """Initialize the Converter."""

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

    def print_remaining_credit(self, service_credit):
        if launcher_client_v2.is_cloud():
            remaining_credit = auth_client.get_credit(
                self.token_handler.tokens.access_token, verify_ssl=self.token_handler.verify_ssl
            )
            logger.info(
                f"{service_credit} credits have been consumed. Remaining Credit: {remaining_credit}"
            )

    def initialize_metadata(self, output_dir):
        def create_metadata_with_status(status, error_message=None):
            metadata = ConverterMetadata()
            metadata.status = status
            if error_message:
                logger.error(error_message)
            return metadata

        try:
            metadata = ConverterMetadata()
        except Exception as e:
            error_message = f"An unexpected error occurred during metadata initialization: {e}"
            metadata = create_metadata_with_status(Status.ERROR, error_message)
        except KeyboardInterrupt:
            warning_message = "Conversion task was interrupted by the user."
            metadata = create_metadata_with_status(Status.STOPPED, warning_message)
        finally:
            MetadataHandler.save_metadata(data=metadata, folder_path=output_dir)

        return metadata

    def handle_conversion_error(self, metadata, error_message):
        metadata.status = Status.ERROR
        metadata.update_message(exception_detail=error_message)
        logger.error(f"Conversion task failed due to an error: {error_message}")

        return metadata

    def _download_converted_model(
        self, convert_task: ConvertTask, local_path: str
    ) -> None:
        """Download the converted model with given conversion task or conversion task uuid.

        Args:
            conversion_task (ConvertTask): Launcher Model Object or the uuid of the conversion task.

        Raises:
            e: If an error occurs while getting the conversion task information.
        """

        self.token_handler.validate_token()

        try:
            download_url = launcher_client_v2.converter.download_model_file(
                convert_task_uuid=convert_task.convert_task_id,
                access_token=self.token_handler.tokens.access_token,
            ).data.presigned_download_url

            request.urlretrieve(download_url, local_path)
            logger.info(f"Model downloaded at {Path(local_path)}")

        except Exception as e:
            logger.error(f"Download converted model failed. Error: {e}")
            raise e

    def convert_model(
        self,
        input_model_path: str,
        output_dir: str,
        target_framework: Union[str, Framework],
        target_device_name: Union[str, DeviceName],
        target_data_type: Union[str, DataType] = DataType.FP16,
        target_software_version: Optional[Union[str, SoftwareVersion]] = None,
        input_layer: Optional[InputLayer] = None,
        dataset_path: Optional[str] = None,
        wait_until_done: bool = True,
        sleep_interval: int = 30,
    ) -> ConverterMetadata:
        """Convert a model to the specified framework.

        Args:
            input_model_path (str): The file path where the model is located.
            output_dir (str): The local folder path to save the converted model.
            target_framework (Union[str, Framework]): The target framework name.
            target_device_name (Union[str, DeviceName]): Target device name. Required if target_device is not specified.
            target_data_type (Union[str, DataType]): Data type of the model. Default is DataType.FP16.
            target_software_version (Union[str, SoftwareVersion], optional): Target software version.
                Required if target_device_name is one of the Jetson devices.
            input_layer (InputShape, optional): Target input shape for conversion (e.g., dynamic batch to static batch).
            dataset_path (str, optional): Path to the dataset. Useful for certain conversions.
            wait_until_done (bool): If True, wait for the conversion result before returning the function.
                                If False, request the conversion and return  the function immediately.

        Raises:
            e: If an error occurs during the model conversion.

        Returns:
            ConverterMetadata: Convert metadata.
        """

        FileHandler.check_input_model_path(input_model_path)
        output_dir = FileHandler.create_unique_folder(folder_path=output_dir)
        metadata = self.initialize_metadata(output_dir=output_dir)

        try:
            metadata.input_model_path = Path(input_model_path).resolve().as_posix()
            if metadata.status in [Status.ERROR, Status.STOPPED]:
                return metadata
            default_model_path = FileHandler.get_default_model_path(folder_path=output_dir)
            extension = FileHandler.get_extension(framework=target_framework)

            self.token_handler.validate_token()
            self.check_credit_balance(service_task=ServiceTask.MODEL_CONVERT)

            # Get presigned_model_upload_url
            presigned_url_response = launcher_client_v2.converter.presigned_model_upload_url(
                access_token=self.token_handler.tokens.access_token,
                input_model_path=input_model_path,
            )

            # Upload model_file
            launcher_client_v2.converter.upload_model_file(
                access_token=self.token_handler.tokens.access_token,
                input_model_path=input_model_path,
                presigned_upload_url=presigned_url_response.data.presigned_upload_url,
            )

            # Validate model_file
            validate_model_response = launcher_client_v2.converter.validate_model_file(
                access_token=self.token_handler.tokens.access_token,
                input_model_path=input_model_path,
                ai_model_id=presigned_url_response.data.ai_model_id,
            )

            # Start convert task
            convert_response = launcher_client_v2.converter.start_task(
                access_token=self.token_handler.tokens.access_token,
                input_model_id=presigned_url_response.data.ai_model_id,
                target_device_name=target_device_name,
                target_framework=target_framework,
                data_type=target_data_type,
                input_layer=input_layer if input_layer else validate_model_response.data.detail.input_layers[0],
                software_version=target_software_version,
                dataset_path=dataset_path,
            )

            metadata.model_info = validate_model_response.data.to()
            metadata.convert_task_info = convert_response.data.to(validate_model_response.data.uploaded_file_name)
            MetadataHandler.save_metadata(data=metadata, folder_path=output_dir)

            if wait_until_done:
                while True:
                    self.token_handler.validate_token()
                    convert_response = launcher_client_v2.converter.read_task(
                        access_token=self.token_handler.tokens.access_token,
                        task_id=convert_response.data.convert_task_id,
                    )
                    if convert_response.data.status in [
                        TaskStatusForDisplay.FINISHED,
                        TaskStatusForDisplay.ERROR,
                        TaskStatusForDisplay.TIMEOUT,
                    ]:
                        break

                    time.sleep(sleep_interval)

            available_options = launcher_client_v2.benchmarker.read_framework_options(
                access_token=self.token_handler.tokens.access_token,
                framework=target_framework,
            )
            metadata.available_options.extend(option.to() for option in available_options.data)

            if convert_response.data.status == TaskStatusForDisplay.FINISHED:
                self._download_converted_model(
                    convert_task=convert_response.data,
                    local_path=str(default_model_path.with_suffix(extension)),
                )
                self.print_remaining_credit(service_credit=ServiceTask.MODEL_CONVERT)
                metadata.status = Status.COMPLETED
                metadata.converted_model_path = default_model_path.with_suffix(extension).as_posix()
                logger.info("Conversion task was completed successfully.")
            else:
                metadata = self.handle_conversion_error(metadata, convert_response.data.error_log)

        except Exception as e:
            metadata = self.handle_conversion_error(metadata, e.args[0])
        except KeyboardInterrupt:
            metadata.status = Status.STOPPED
            logger.error("Conversion task was interrupted by the user.")
        finally:
            MetadataHandler.save_metadata(data=metadata, folder_path=output_dir)

        return metadata

    def get_conversion_task(self, conversion_task_id: str) -> ConvertTask:
        """Get the conversion task information with given conversion task uuid.

        Args:
            conversion_task_id (str): Convert task UUID of the convert task.

        Raises:
            e: If an error occurs during the model conversion.

        Returns:
            ConversionTask: Model conversion task dictionary.
        """

        self.token_handler.validate_token()

        response = launcher_client_v2.converter.read_task(
            access_token=self.token_handler.tokens.access_token,
            task_id=conversion_task_id,
        )
        return response.data
