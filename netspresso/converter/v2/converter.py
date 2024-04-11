import time
from pathlib import Path
from typing import Union, Optional, Dict
from urllib import request

from loguru import logger

from netspresso.clients.auth import TokenHandler
from netspresso.clients.auth.response_body import UserResponse
from netspresso.clients.launcher import launcher_client_v2
from netspresso.clients.launcher.v2.schemas.task.convert.response_body import (
    ConvertTask,
)
from netspresso.enums import TaskStatusForDisplay
from netspresso.clients.launcher.v2.schemas import InputLayer, ResponseConvertTaskItem
from netspresso.enums import Framework, DeviceName, DataType, SoftwareVersion
from netspresso.utils import FileHandler


class ConverterV2:
    def __init__(self, token_handler: TokenHandler, user_info: UserResponse):
        """Initialize the Converter."""

        self.token_handler = token_handler
        self.user_info = user_info

    def _download_converted_model(
        self, conversion_task: ConvertTask, local_path: str
    ) -> None:
        """Download the converted model with given conversion task or conversion task uuid.

        Args:
            conversion_task (ConvertTask): Launcher Model Object or the uuid of the conversion task.

        Raises:
            e: If an error occurs while getting the conversion task information.
        """

        try:
            if conversion_task.status == TaskStatusForDisplay.ERROR:
                raise FileNotFoundError(
                    "The conversion is Failed. There is no file available for download."
                )
            if conversion_task.status != TaskStatusForDisplay.FINISHED:
                raise FileNotFoundError(
                    "The conversion is in progress. There is no file available for download at the moment."
                )

            download_url = launcher_client_v2.converter.download_model_file(
                convert_task_uuid=conversion_task.convert_task_id,
                access_token=self.token_handler.tokens.access_token,
            )
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
    ) -> ConvertTask:
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
            ResponseConvertTaskItem: Model conversion task result.
        """

        FileHandler.check_input_model_path(input_model_path)

        self.token_handler.validate_token()

        default_model_path, extension = FileHandler.get_path_and_extension(
            folder_path=output_dir, framework=target_framework
        )
        FileHandler.create_unique_folder(folder_path=output_dir)
        # metadata = MetadataHandler.init_metadata(folder_path=output_dir, task_type=TaskType.CONVERT)

        # GET presigned_model_upload_url
        presigned_url_response = (
            launcher_client_v2.converter.presigned_model_upload_url(
                access_token=self.token_handler.tokens.access_token,
                input_model_path=input_model_path,
            )
        )

        # UPLOAD model_file
        model_upload_response = launcher_client_v2.converter.upload_model_file(
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

        # START convert task
        response = launcher_client_v2.converter.start_task(
            access_token=self.token_handler.tokens.access_token,
            input_model_id=presigned_url_response.data.ai_model_id,
            target_device_name=target_device_name,
            target_framework=target_framework,
            data_type=target_data_type,
            input_layer=input_layer,
            software_version=target_software_version,
            dataset_path=dataset_path,
        )

        if wait_until_done:
            while True:
                # Poll Convert Task status
                response = launcher_client_v2.converter.read_task(
                    access_token=self.token_handler.tokens.access_token,
                    task_id=response.data.convert_task_id,
                )
                if response.data.status in [
                    TaskStatusForDisplay.FINISHED,
                    TaskStatusForDisplay.CANCELLED,
                ]:
                    break
                time.sleep(3)

        self._download_converted_model(
            conversion_task=response.data,
            local_path=str(default_model_path.with_suffix(extension)),
        )

        # metadata.update_converted_model_path(
        #     converted_model_path=default_model_path.with_suffix(extension).as_posix()
        # )
        # metadata.update_model_info(
        #     data_type=model.data_type,
        #     framework=model.framework,
        #     input_shape=model.input_shape,
        # )
        # metadata.update_convert_info(
        #     target_framework=conversion_task.target_framework,
        #     target_device_name=conversion_task.target_device_name,
        #     data_type=conversion_task.data_type,
        #     software_version=conversion_task.software_version,
        #     model_file_name=conversion_task.model_file_name,
        #     convert_task_uuid=conversion_task.convert_task_uuid,
        #     input_model_uuid=conversion_task.input_model_uuid,
        #     output_model_uuid=conversion_task.output_model_uuid,
        # )
        # metadata.update_status(status=Status.COMPLETED)
        # metadata.update_available_devices(converter_uploaded_model.available_devices)
        # MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_dir)

        return response.data

    def get_conversion_task(self, conversion_task_id: str) -> ConvertTask:
        response = launcher_client_v2.converter.read_task(
            access_token=self.token_handler.tokens.access_token,
            task_id=conversion_task_id,
        )
        return response.data

    def __generate_metadata(self):
        pass
