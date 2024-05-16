from pathlib import Path
import sys
from typing import Dict, List, Optional
from urllib import request

from loguru import logger

from netspresso.clients.auth import TokenHandler, auth_client
from netspresso.clients.compressor import compressor_client_v2
from netspresso.clients.compressor.v2.schemas import (
    RequestCreateModel,
    RequestUploadModel,
    RequestValidateModel,
    UploadFile,
    ModelBase,
    Options,
    RecommendationOptions,
    ResponseCompressionItem,
    ResponseCompression,
    RequestAvailableLayers,
    ResponseSelectMethod,
    RequestAutomaticCompressionParams,
    RequestCreateCompression,
    RequestCreateRecommendation,
    RequestUpdateCompression,
)
from netspresso.clients.launcher import launcher_client_v2
from netspresso.compressor.utils.file import read_file_bytes
from netspresso.compressor.utils.onnx import export_onnx
from netspresso.enums import CompressionMethod, Framework, Module, RecommendationMethod, ServiceCredit, Status, TaskType
from netspresso.utils import FileHandler
from netspresso.utils.metadata import MetadataHandler


class CompressorV2:
    def __init__(self, token_handler: TokenHandler) -> None:
        """Initialize the Compressor."""

        self.token_handler = token_handler

    def check_credit_balance(self, service_credit):
        # TODO: Create a base class and move functions
        current_credit = auth_client.get_credit(
            access_token=self.token_handler.tokens.access_token,
            verify_ssl=self.token_handler.verify_ssl
        )
        service_name = service_credit.name.replace("_", " ").lower()
        if current_credit < service_credit:
            sys.exit(f"Your current balance of {current_credit} credits is insufficient to complete the task. \n{service_credit} credits are required for one {service_name} task. \nFor additional credit, please contact us at netspresso@nota.ai.")

    def upload_model(
        self,
        input_model_path: str,
        input_shapes: List[Dict[str, int]] = None,
        framework: Framework = Framework.PYTORCH,
    ) -> ModelBase:

        FileHandler.check_input_model_path(input_model_path)

        self.token_handler.validate_token()

        try:
            logger.info("Uploading Model...")

            object_name = Path(input_model_path).stem

            create_model_request = RequestCreateModel(object_name=object_name)
            create_model_response = compressor_client_v2.create_model(
                request_data=create_model_request,
                access_token=self.token_handler.tokens.access_token,
                verify_ssl=self.token_handler.verify_ssl,
            )

            file_content = read_file_bytes(file_path=input_model_path)
            upload_model_request = RequestUploadModel(url=create_model_response.data.presigned_url)
            file = UploadFile(file_name=object_name, file_content=file_content)
            upload_model_response = compressor_client_v2.upload_model(
                request_data=upload_model_request,
                file=file,
                access_token=self.token_handler.tokens.access_token,
                verify_ssl=self.token_handler.verify_ssl,
            )

            if not upload_model_response:
                # TODO: Confirm upload success
                pass

            validate_model_request = RequestValidateModel(framework=framework, input_layers=input_shapes)
            validate_model_response = compressor_client_v2.validate_model(
                ai_model_id=create_model_response.data.ai_model_id,
                request_data=validate_model_request,
                access_token=self.token_handler.tokens.access_token,
                verify_ssl=self.token_handler.verify_ssl,
            )

            model_info = validate_model_response.data

            logger.info(f"Upload model successfully. Model ID: {model_info.ai_model_id}")

            return model_info

        except Exception as e:
            logger.error(f"Upload model failed. Error: {e}")
            raise e

    def get_model(self, model_id: str) -> ModelBase:
        self.token_handler.validate_token()

        try:
            logger.info("Getting model...")
            read_model_response = compressor_client_v2.read_model(
                ai_model_id=model_id,
                access_token=self.token_handler.tokens.access_token,
                verify_ssl=self.token_handler.verify_ssl,
            )
            model_info = read_model_response.data

            logger.info("Get model successfully.")

            return model_info

        except Exception as e:
            logger.error(f"Get model failed. Error: {e}")
            raise e

    def download_model(self, model_id: str, local_path: str) -> None:
        self.token_handler.validate_token()

        try:
            logger.info("Downloading model...")
            download_link = compressor_client_v2.download_model(
                ai_model_id=model_id,
                access_token=self.token_handler.tokens.access_token,
                verify_ssl=self.token_handler.verify_ssl,
            )
            request.urlretrieve(download_link.data.presigned_url, local_path)
            logger.info(f"Model downloaded at {Path(local_path)}")

        except Exception as e:
            logger.error(f"Download model failed. Error: {e}")
            raise e

    def select_compression_method(
        self,
        model_id: str,
        compression_method: CompressionMethod,
        options: Optional[Options] = Options(),
    ) -> ResponseSelectMethod:
        self.token_handler.validate_token()

        try:
            logger.info("Selecting compression method...")

            # 이거를 backend에서 처리할 지 고민
            # if model.framework == Framework.PYTORCH and compression_method == CompressionMethod.PR_NN:
            #     raise Exception("The Nuclear Norm is only supported in the TensorFlow-Keras framework.")

            get_available_layers_request = RequestAvailableLayers(
                compression_method=compression_method,
                options=options,
            )
            get_available_layers_response = compressor_client_v2.get_available_layers(
                ai_model_id=model_id,
                request_data=get_available_layers_request,
                access_token=self.token_handler.tokens.access_token,
                verify_ssl=self.token_handler.verify_ssl,
            )

            available_layers_info = get_available_layers_response.data

            logger.info("Select compression method successfully.")

            return available_layers_info

        except Exception as e:
            logger.error(f"Select compression method failed. Error: {e}")
            raise e

    def get_compression(self, compression_id: str) -> ResponseCompression:
        self.token_handler.validate_token()

        try:
            logger.info("Getting compression...")
            read_compression_response = compressor_client_v2.read_compression(
                compression_id=compression_id,
                access_token=self.token_handler.tokens.access_token,
                verify_ssl=self.token_handler.verify_ssl,
            )

            compression_info = read_compression_response.data

            logger.info("Get compression successfully.")

            return compression_info

        except Exception as e:
            logger.error(f"Get compression failed. Error: {e}")
            raise e

    def compress_model(
        self,
        compression: ResponseSelectMethod,
        output_dir: str,
        dataset_path: Optional[str] = None,
    ) -> Dict:
        self.token_handler.validate_token()

        try:
            logger.info("Compressing model...")

            model_info = self.get_model(compression.input_model_id)

            output_dir = FileHandler.create_unique_folder(folder_path=output_dir)
            default_model_path, extension = FileHandler.get_path_and_extension(
                folder_path=output_dir, framework=model_info.detail.framework
            )
            metadata = MetadataHandler.init_metadata(folder_path=output_dir, task_type=TaskType.COMPRESS)

            self.check_credit_balance(service_credit=ServiceCredit.ADVANCED_COMPRESSION)

            create_compression_request = RequestCreateCompression(
                ai_model_id=compression.input_model_id,
                compression_method=compression.compression_method,
                options=compression.options,
            )
            create_compression_response = compressor_client_v2.create_compression(
                request_data=create_compression_request,
                access_token=self.token_handler.tokens.access_token,
                verify_ssl=self.token_handler.verify_ssl
            )

            # if dataset_path and compression.compression_method == CompressionMethod.PR_NN:
            #     self.__upload_dataset(model_id=compression.original_model_id, dataset_path=dataset_path)

            for available_layers in compression.available_layers:
                if available_layers.values:
                    available_layers.use = True

            update_compression_request = RequestUpdateCompression(
                available_layers=compression.available_layers,
                options=compression.options,
            )
            update_compression_response = compressor_client_v2.compress_model(
                compression_id=create_compression_response.data.compression_id,
                request_data=update_compression_request,
                access_token=self.token_handler.tokens.access_token,
                verify_ssl=self.token_handler.verify_ssl
            )

            compression_info = update_compression_response.data

            self.download_model(
                model_id=compression_info.input_model_id,
                local_path=default_model_path.with_suffix(extension),
            )

            # TODO: For available devices
            # converter_uploaded_model = self._get_available_devices(compressed_model, default_model_path)

            logger.info(f"Compress model successfully. Compressed Model ID: {compression_info.input_model_id}")

            if compressor_client_v2.is_cloud():
                remaining_credit = auth_client.get_credit(
                    self.token_handler.tokens.access_token, verify_ssl=self.token_handler.verify_ssl
                )
                logger.info(
                    f"{ServiceCredit.ADVANCED_COMPRESSION} credits have been consumed. Remaining Credit: {remaining_credit}"
                )

            # metadata.update_compressed_model_path(
            #     compressed_model_path=default_model_path.with_suffix(extension).as_posix()
            # )
            # if compressed_model.framework in [Framework.PYTORCH, Framework.ONNX]:
            #     metadata.update_compressed_onnx_model_path(
            #         compressed_onnx_model_path=default_model_path.with_suffix(".onnx").as_posix()
            #     )
            # metadata.update_model_info(
            #     task=model_info.task,
            #     framework=model_info.framework,
            #     input_shapes=model_info.input_shapes,
            # )
            # metadata.update_compression_info(
            #     method=compression.compression_method,
            #     options=compression.options,
            #     layers=compression.available_layers,
            # )
            # metadata.update_results(model=model_info, compressed_model=compressed_model)
            # metadata.update_status(status=Status.COMPLETED)
            # metadata.update_available_devices(converter_uploaded_model.available_devices)
            # MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_dir)

            return compression_info

        except Exception as e:
            logger.error(f"Compress model failed. Error: {e}")
            metadata.update_status(status=Status.ERROR)
            MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_dir)
            raise e

        except KeyboardInterrupt:
            metadata.update_status(status=Status.STOPPED)
            MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_dir)
    
    def recommendation_compression(
        self,
        compression_method: CompressionMethod,
        recommendation_method: RecommendationMethod,
        recommendation_ratio: float,
        input_model_path: str,
        output_dir: str,
        input_shapes: List[Dict[str, int]],
        framework: Framework = Framework.PYTORCH,
        options: RecommendationOptions = RecommendationOptions(),
        dataset_path: Optional[str] = None,
    ) -> Dict:
        FileHandler.check_input_model_path(input_model_path)

        self.token_handler.validate_token()

        try:
            logger.info("Compressing recommendation-based model...")

            output_dir = FileHandler.create_unique_folder(folder_path=output_dir)
            default_model_path, extension = FileHandler.get_path_and_extension(folder_path=output_dir, framework=framework)
            metadata = MetadataHandler.init_metadata(folder_path=output_dir, task_type=TaskType.COMPRESS)

            self.check_credit_balance(service_credit=ServiceCredit.ADVANCED_COMPRESSION)

            if framework == Framework.PYTORCH and compression_method == CompressionMethod.PR_NN:
                raise Exception("The Nuclear Norm is only supported in the TensorFlow-Keras framework.")

            model = self.upload_model(
                framework=framework,
                input_model_path=input_model_path,
                input_shapes=input_shapes,
            )

            create_compression_request = RequestCreateCompression(
                ai_model_id=model.ai_model_id,
                compression_method=compression_method,
                options=options,
            )
            create_compression_response = compressor_client_v2.create_compression(
                request_data=create_compression_request,
                access_token=self.token_handler.tokens.access_token,
                verify_ssl=self.token_handler.verify_ssl
            )

            # if dataset_path and compression_method == CompressionMethod.PR_NN:
            #     self.__upload_dataset(model_id=model.model_id, dataset_path=dataset_path)

            logger.info("Calculating recommendation values...")
            create_recommendation_request = RequestCreateRecommendation(
                recommendation_method=recommendation_method,
                recommendation_ratio=recommendation_ratio,
                options=options,                
            )
            create_recommendation_response = compressor_client_v2.create_recommendation(
                compression_id=create_compression_response.data.compression_id,
                request_data=create_recommendation_request,
                access_token=self.token_handler.tokens.access_token,
                verify_ssl=self.token_handler.verify_ssl,
            )

            for layer in create_recommendation_response.data.available_layers:
                layer.use = True

            logger.info("Compressing model...")
            update_compression_request = RequestUpdateCompression(
                available_layers=create_recommendation_response.data.available_layers,
                options=options,
            )
            update_compression_response = compressor_client_v2.compress_model(
                compression_id=create_compression_response.data.compression_id,
                request_data=update_compression_request,
                access_token=self.token_handler.tokens.access_token,
                verify_ssl=self.token_handler.verify_ssl
            )

            compression_info = update_compression_response.data

            self.download_model(
                model_id=compression_info.input_model_id,
                local_path=default_model_path.with_suffix(extension),
            )

            # TODO: For available devices
            # converter_uploaded_model = self._get_available_devices(compressed_model, default_model_path)

            logger.info(f"Recommendation compression successfully. Compressed Model ID: {compression_info.input_model_id}")

            if compressor_client_v2.is_cloud():
                remaining_credit = auth_client.get_credit(
                    self.token_handler.tokens.access_token, verify_ssl=self.token_handler.verify_ssl
                )
                logger.info(
                    f"{ServiceCredit.ADVANCED_COMPRESSION} credits have been consumed. Remaining Credit: {remaining_credit}"
                )

            # metadata.update_compressed_model_path(
            #     compressed_model_path=default_model_path.with_suffix(extension).as_posix()
            # )
            # if compressed_model.framework in [Framework.PYTORCH, Framework.ONNX]:
            #     metadata.update_compressed_onnx_model_path(
            #         compressed_onnx_model_path=default_model_path.with_suffix(".onnx").as_posix()
            #     )
            # metadata.update_model_info(task=model.task, framework=framework, input_shapes=input_shapes)
            # metadata.update_compression_info(
            #     method=_compression_info.compression_method,
            #     ratio=recommendation_ratio,
            #     options=_compression_info.options,
            #     layers=_compression_info.available_layers,
            # )
            # metadata.update_results(model=model, compressed_model=compressed_model)
            # metadata.update_status(status=Status.COMPLETED)
            # metadata.update_available_devices(converter_uploaded_model.available_devices)
            # MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_dir)
                
            print(compression_info)

            return compression_info

        except Exception as e:
            logger.error(f"Recommendation compression failed. Error: {e}")
            metadata.update_status(status=Status.ERROR)
            MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_dir)
            raise e

        except KeyboardInterrupt:
            metadata.update_status(status=Status.STOPPED)
            MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_dir)

    def automatic_compression(
        self,
        input_model_path: str,
        output_dir: str,
        input_shapes: List[Dict[str, int]],
        framework: Framework = Framework.PYTORCH,
        compression_ratio: float = 0.5,
    ) -> Dict:
        FileHandler.check_input_model_path(input_model_path)

        self.token_handler.validate_token()

        try:
            logger.info("Compressing automatic-based model...")

            output_dir = FileHandler.create_unique_folder(folder_path=output_dir)
            default_model_path, extension = FileHandler.get_path_and_extension(
                folder_path=output_dir, framework=framework
            )
            metadata = MetadataHandler.init_metadata(folder_path=output_dir, task_type=TaskType.COMPRESS)

            self.check_credit_balance(service_credit=ServiceCredit.AUTOMATIC_COMPRESSION)

            model_info = self.upload_model(framework=framework, input_model_path=input_model_path, input_shapes=input_shapes)
           
            automatic_compression_request = RequestAutomaticCompressionParams(compression_ratio=compression_ratio)
            
            logger.info("Compressing model...")

            automatic_compression_response = compressor_client_v2.compress_model_with_automatic(
                ai_model_id=model_info.ai_model_id,
                request_data=automatic_compression_request,
                access_token=self.token_handler.tokens.access_token,
                verify_ssl=self.token_handler.verify_ssl,
            )

            compression_info = automatic_compression_response.data

            self.download_model(
                model_id=compression_info.input_model_id,
                local_path=default_model_path.with_suffix(extension),
            )

            # converter_uploaded_model = self._get_available_devices(compressed_model, default_model_path)

            logger.info(f"Automatic compression successfully. Compressed Model ID: {compression_info.input_model_id}")

            if compressor_client_v2.is_cloud():
                remaining_credit = auth_client.get_credit(
                    self.token_handler.tokens.access_token, verify_ssl=self.token_handler.verify_ssl
                )
                logger.info(
                    f"{ServiceCredit.AUTOMATIC_COMPRESSION} credits have been consumed. Remaining Credit: {remaining_credit}"
                )

            # metadata.update_compressed_model_path(
            #     compressed_model_path=default_model_path.with_suffix(extension).as_posix()
            # )
            # if compressed_model_info.detail.framework in [Framework.PYTORCH, Framework.ONNX]:
            #     metadata.update_compressed_onnx_model_path(
            #         compressed_onnx_model_path=default_model_path.with_suffix(".onnx").as_posix()
            #     )
            # metadata.update_model_info(framework=framework, input_shapes=input_shapes)
            # metadata.update_compression_info(
            #     method=compression_info.compression_method,
            #     ratio=compression_ratio,
            #     options=compression_info.options,
            #     layers=compression_info.available_layers,
            # )
            # metadata.update_results(model=model, compressed_model=compressed_model)
            # metadata.update_status(status=Status.COMPLETED)
            # # metadata.update_available_devices(converter_uploaded_model.available_devices)
            # MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_dir)

            return compression_info

        except Exception as e:
            logger.error(f"Automatic compression failed. Error: {e}")
            metadata.update_status(status=Status.ERROR)
            MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_dir)
            raise e

        except KeyboardInterrupt:
            metadata.update_status(status=Status.STOPPED)
            MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_dir)
