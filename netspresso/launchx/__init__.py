from typing import Dict, List, Union

from loguru import logger
from urllib import request

from netspresso.client import BaseClient, validate_token
from netspresso.launchx.schemas import LaunchXFunction, ModelFramework, TaskStatus, DataType
from netspresso.launchx.schemas.model import Model, ConversionTask, BenchmarkTask, InputShape, TargetDevice
from netspresso.launchx.client import LaunchXAPIClient

class LaunchXClient(BaseClient):
    target_function: LaunchXFunction = LaunchXFunction.GENERAL

    def __init__(self, *args, **kwargs):
        """Initialize the Model Compressor.

        Args:
            email (str): The email address for a user account.
            password (str): The password for a user account.
            user_session (SessionClient): The SessionClient object.

        Available constructors: 
            LaunchXClient(email='USER_EMAIL',password='PASSWORD')
            LaunchXClient(user_session=SessionClient(email='USER_EMAIL',password='PASSWORD')
        """
        super().__init__(*args, **kwargs)
        self.client = LaunchXAPIClient(user_sessoin=self.user_session)

    @validate_token
    def upload_model(self, model_file_path: str) -> Model:
        return self.client.upload_model(model_file_path=model_file_path,
                                        target_function=self.__class__.target_function)
    @validate_token
    def download_model(self, model: Union[str, Model]):
        pass

class LaunchXConverter(LaunchXClient):
    target_function: LaunchXFunction = LaunchXFunction.CONVERT

    @validate_token
    def convert_model(self, model: Union[str, Model],
                      input_shape: InputShape,
                      target_framework: Union[str, ModelFramework],
                      target_device: TargetDevice) -> ConversionTask:
        model_uuid = model
        if type(model) is Model:
            model_uuid = model.model_uuid
            if input_shape is None and model.input_shape is not None:
                input_shape = model.input_shape
            if target_framework is None and model.framework is not None:
                target_framework = model.framework

        logger.info(f"Converting Model for {target_device.device_name} ({target_framework})")
            
        return self.client.convert_model(model_uuid=model_uuid,
                                         input_shape=input_shape,
                                         target_framework=target_framework,
                                         target_device=target_device.device_name,
                                         software_version=target_device.software_version)

    @validate_token
    def get_conversion_task(self, conversion_task: Union[str, ConversionTask]) -> ConversionTask:
        conversion_task_uuid = None
        if type(conversion_task) is str:
            conversion_task_uuid = conversion_task
        elif type(conversion_task) is ConversionTask:
            conversion_task_uuid = conversion_task.convert_task_uuid
        else:
            raise NotImplementedError("There is no avaliable function for given paremeter. conversion_task should be uuid string or ModelConversion object.")
        return self.client.get_conversion_task(conversion_task_uuid=conversion_task_uuid)

    @validate_token
    def download_converted_model(self, conversion_task: Union[str, ConversionTask], dst: str):
        conversion_task_uuid = None
        if type(conversion_task) is str:
            conversion_task_uuid = conversion_task
        elif type(conversion_task) is ConversionTask:
            conversion_task_uuid = conversion_task.convert_task_uuid

        conversion_result: ConversionTask = self.get_conversion_task(conversion_task_uuid)
        if conversion_result.status is TaskStatus.ERROR:
            raise FileNotFoundError("Conversion is Failed. No file to download.")
        if conversion_result.status is not TaskStatus.FINISHED:
            raise FileNotFoundError("Conversion is running. No file to download yet.")

        download_url = self.client.get_converted_model(conversion_task_uuid=conversion_result.convert_task_uuid)
        request.urlretrieve(download_url, dst)
        logger.info(f"Download model successful. Local Path: {dst}")

class LaunchXBenchmarker(LaunchXClient):
    target_function: LaunchXFunction = LaunchXFunction.BENCHMARK
    @validate_token
    def benchmark_model(self, model: Union[str, Model, ConversionTask],
                        target_device: TargetDevice = None,
                        data_type: DataType = DataType.FP16) -> BenchmarkTask:
        model_uuid = None
        benchmark_data_type = None
        target_device_name = target_device.device_name if target_device is not None else None
        target_software_version = target_device.software_version if target_device is not None else None

        if type(model) is str:
            model_uuid = model
            benchmark_data_type = data_type
        elif type(model) is Model:
            model_uuid = model.model_uuid
            benchmark_data_type = model.data_type
        elif type(model) is ConversionTask: 
            model_uuid = model.output_model_uuid
            benchmark_data_type = model.data_type
            if target_device_name is None:
                target_device_name = model.target_device_name
            if target_software_version is None:
                target_software_version = model.software_version
        if target_device_name is None:
            raise NotImplementedError("There is no avaliable function for given paremeter. Please specify target device.")

        model_benchmark: BenchmarkTask = self.client.benchmark_model(model_uuid=model_uuid,
                                                                      target_device=target_device_name,
                                                                      data_type=benchmark_data_type,
                                                                      software_version=target_software_version)

        return model_benchmark
    
    @validate_token
    def get_benchmark_task(self, benchmark_task: Union[str, BenchmarkTask]) -> BenchmarkTask:
        task_uuid = None
        if type(benchmark_task) is str:
            task_uuid = benchmark_task
        elif type(benchmark_task) is BenchmarkTask:
            task_uuid = benchmark_task.benchmark_task_uuid
        else:
            raise NotImplementedError("There is no avaliable function for given paremeter. benchmark_task should be uuid string or ModelBenchmark object.")

        return self.client.get_benchmark(benchmark_task_uuid=task_uuid)