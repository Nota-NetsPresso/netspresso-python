import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Union
from urllib import request

from loguru import logger

from netspresso.clients.auth import TokenHandler, auth_client
from netspresso.clients.auth.response_body import UserResponse
from netspresso.clients.launcher import launcher_client_v2
from netspresso.clients.launcher.v2.schemas import InputLayer, ResponseConvertTaskItem
from netspresso.clients.launcher.v2.schemas.task.convert.response_body import (
    ConvertTask,
)
from netspresso.enums import (
    DataType,
    DeviceName,
    Framework,
    ServiceCredit,
    SoftwareVersion,
    Status,
    TaskStatusForDisplay,
)
from netspresso.metadata.converter import ConverterMetadata
from netspresso.utils import FileHandler, check_credit_balance
from netspresso.utils.metadata import MetadataHandler


class ConverterV2:
    def __init__(self, token_handler: TokenHandler, user_info: UserResponse):
        """Initialize the Converter."""

        self.token_handler = token_handler
        self.user_info = user_info

    def _download_converted_model(
        self, convert_task: ConvertTask, local_path: str
    ) -> None:
        """Download the converted model with given conversion task or conversion task uuid.

        Args:
            conversion_task (ConvertTask): Launcher Model Object or the uuid of the conversion task.

        Raises:
            e: If an error occurs while getting the conversion task information.
        """

        try:
            if convert_task.status == TaskStatusForDisplay.ERROR:
                raise FileNotFoundError(
                    "The conversion is Failed. There is no file available for download."
                )
            if convert_task.status != TaskStatusForDisplay.FINISHED:
                raise FileNotFoundError(
                    "The conversion is in progress. There is no file available for download at the moment."
                )

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

        self.token_handler.validate_token()

        default_model_path, extension = FileHandler.get_path_and_extension(
            folder_path=output_dir, framework=target_framework
        )
        output_dir = FileHandler.create_unique_folder(folder_path=output_dir)
        converter_metadata = ConverterMetadata()
        converter_metadata.input_model_path = input_model_path
        MetadataHandler.save_json(data=asdict(converter_metadata), folder_path=output_dir)

        try:
            current_credit = auth_client.get_credit(
                self.token_handler.tokens.access_token,
                verify_ssl=self.token_handler.verify_ssl,
            )
            check_credit_balance(
                user_credit=current_credit, service_credit=ServiceCredit.MODEL_CONVERT
            )

            # GET presigned_model_upload_url
            presigned_url_response = (
                launcher_client_v2.converter.presigned_model_upload_url(
                    access_token=self.token_handler.tokens.access_token,
                    input_model_path=input_model_path,
                )
            )

            # UPLOAD model_file
            launcher_client_v2.converter.upload_model_file(
                access_token=self.token_handler.tokens.access_token,
                input_model_path=input_model_path,
                presigned_upload_url=presigned_url_response.data.presigned_upload_url,
            )

            # VALIDATE model_file
            validate_model_response = launcher_client_v2.converter.validate_model_file(
                access_token=self.token_handler.tokens.access_token,
                input_model_path=input_model_path,
                ai_model_id=presigned_url_response.data.ai_model_id,
            )

            input_model_info = validate_model_response.data

            # START convert task
            response = launcher_client_v2.converter.start_task(
                access_token=self.token_handler.tokens.access_token,
                input_model_id=presigned_url_response.data.ai_model_id,
                target_device_name=target_device_name,
                target_framework=target_framework,
                data_type=target_data_type,
                input_layer=input_layer if input_layer else input_model_info.detail.input_layers[0],
                software_version=target_software_version,
                dataset_path=dataset_path,
            )

            converter_metadata.model_info = input_model_info.to()
            converter_metadata.convert_task_info = response.data.to(input_model_info.uploaded_file_name)
            MetadataHandler.save_json(data=asdict(converter_metadata), folder_path=output_dir)

            if wait_until_done:
                while True:
                    # Poll Convert Task status
                    response = launcher_client_v2.converter.read_task(
                        access_token=self.token_handler.tokens.access_token,
                        task_id=response.data.convert_task_id,
                    )
                    if response.data.status in [
                        TaskStatusForDisplay.FINISHED,
                        TaskStatusForDisplay.ERROR,
                        TaskStatusForDisplay.TIMEOUT,
                    ]:
                        break
                    self.token_handler.validate_token()
                    time.sleep(3)

            self._download_converted_model(
                convert_task=response.data,
                local_path=str(default_model_path.with_suffix(extension)),
            )

            if launcher_client_v2.is_cloud():
                remaining_credit = auth_client.get_credit(
                    self.token_handler.tokens.access_token,
                    self.token_handler.verify_ssl,
                )
                logger.info(
                    f"{ServiceCredit.MODEL_BENCHMARK} credits have been consumed. Remaining Credit: {remaining_credit}"
                )

            convert_task = response.data

            available_options = launcher_client_v2.benchmarker.read_framework_options(
                access_token=self.token_handler.tokens.access_token,
                framework=target_framework,
            ).data

            if convert_task.status == TaskStatusForDisplay.FINISHED:
                converter_metadata.status = Status.COMPLETED
                logger.info("Convert task successfully completed.")
            else:
                converter_metadata.status = Status.ERROR
                logger.info("Convert task failed with an error.")

            converter_metadata.converted_model_path = str(default_model_path.with_suffix(extension))
            for available_option in available_options:
                converter_metadata.available_options.append(available_option.to())

            MetadataHandler.save_json(
                data=asdict(converter_metadata), folder_path=output_dir
            )

            return converter_metadata

        except Exception as e:
            logger.error(f"Convert failed. Error: {e}")
            converter_metadata.status = Status.ERROR
            MetadataHandler.save_json(
                data=asdict(converter_metadata), folder_path=output_dir
            )
            raise e

        except KeyboardInterrupt:
            converter_metadata.status = Status.STOPPED
            MetadataHandler.save_json(
                data=asdict(converter_metadata), folder_path=output_dir
            )

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
