import sys
from pathlib import Path
from typing import Dict, List, Optional
from urllib import request

from loguru import logger

from netspresso.clients.auth import TokenHandler, auth_client
from netspresso.clients.compressor import compressor_client_v2
from netspresso.clients.compressor.v2.schemas import (
    ModelBase,
    Options,
    RecommendationOptions,
    RequestAutomaticCompressionParams,
    RequestAvailableLayers,
    RequestCreateCompression,
    RequestCreateModel,
    RequestCreateRecommendation,
    RequestUpdateCompression,
    RequestUploadModel,
    RequestValidateModel,
    ResponseCompression,
    ResponseCompressionItem,
    ResponseSelectMethod,
    UploadFile,
)
from netspresso.clients.launcher import launcher_client_v2
from netspresso.compressor.utils.file import read_file_bytes
from netspresso.compressor.utils.onnx import export_onnx
from netspresso.enums import CompressionMethod, Framework, Module, RecommendationMethod, ServiceCredit, Status, TaskType
from netspresso.metadata.compressor import CompressorMetadata
from netspresso.utils import FileHandler
from netspresso.utils.metadata import MetadataHandler


class CompressorV2:
    def __init__(self, token_handler: TokenHandler) -> None:
        """Initialize the Compressor."""

        self.token_handler = token_handler

    def check_credit_balance(self, service_credit):
        current_credit = auth_client.get_credit(
            access_token=self.token_handler.tokens.access_token,
            verify_ssl=self.token_handler.verify_ssl
        )
        service_name = service_credit.name.replace("_", " ").lower()
        if current_credit < service_credit:
            sys.exit(f"Your current balance of {current_credit} credits is insufficient to complete the task. \n{service_credit} credits are required for one {service_name} task. \nFor additional credit, please contact us at netspresso@nota.ai.")

    def print_remaining_credit(self):
        if compressor_client_v2.is_cloud():
            remaining_credit = auth_client.get_credit(
                self.token_handler.tokens.access_token, verify_ssl=self.token_handler.verify_ssl
            )
            logger.info(
                f"{ServiceCredit.ADVANCED_COMPRESSION} credits have been consumed. Remaining Credit: {remaining_credit}"
            )

    def create_metadata(self, folder_path) -> CompressorMetadata:
        metadata = CompressorMetadata()
        MetadataHandler.save_json(data=metadata.asdict(), folder_path=folder_path)

        return metadata

    def _get_available_options(self, compressed_model_info, default_model_path: str):
        if compressed_model_info.detail.framework in [Framework.PYTORCH, Framework.ONNX]:
            export_onnx(default_model_path, compressed_model_info.detail.input_layers)
            options_response = launcher_client_v2.converter.read_framework_options(
                access_token=self.token_handler.tokens.access_token,
                framework=Framework.ONNX,
            )
        else:
            options_response = launcher_client_v2.converter.read_framework_options(
                access_token=self.token_handler.tokens.access_token,
                framework=Framework.TENSORFLOW_KERAS,
            )

        available_options = options_response.data

        return available_options

    def upload_model(
        self,
        input_model_path: str,
        input_shapes: List[Dict[str, int]] = None,
        framework: Framework = Framework.PYTORCH,
    ) -> ModelBase:
        """Upload a model for compression.

        Args:
            input_model_path (str): The file path where the model is located.
            input_shapes (List[Dict[str, int]], optional): Input shapes of the model. Defaults to [].
            framework (Framework): The framework of the model.

        Raises:
            e: If an error occurs while uploading the model.

        Returns:
            ModelBase: Uploaded model object.
        """

        self.token_handler.validate_token()

        try:
            logger.info("Uploading Model...")

            FileHandler.check_input_model_path(input_model_path)

            object_name = Path(input_model_path).name

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
                raise Exception("Upload model failed.")

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
        """Select a compression method for a model.

        Args:
            model_id (str): The ID of the model.
            compression_method (CompressionMethod): The selected compression method.
            options(Options, optional): The options for pruning method.

        Raises:
            e: If an error occurs while selecting the compression method.

        Returns:
            ResponseSelectMethod: The compression information for the selected compression method.
        """

        self.token_handler.validate_token()

        try:
            logger.info("Selecting compression method...")

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

    def upload_dataset(self, compression_id: str, dataset_path: str) -> None:
        self.token_handler.validate_token()

        try:
            logger.info("Uploading dataset...")
            file_content = read_file_bytes(file_path=dataset_path)
            object_name = Path(dataset_path).name
            file = UploadFile(file_name=object_name, file_content=file_content)
            compressor_client_v2.upload_dataset(
                compression_id=compression_id,
                file=file,
                access_token=self.token_handler.tokens.access_token,
                verify_ssl=self.token_handler.verify_ssl,
            )
            logger.info("Upload dataset successfully.")

        except Exception as e:
            logger.error(f"Upload dataset failed. Error: {e}")
            raise e

    def compress_model(
        self,
        compression: ResponseSelectMethod,
        output_dir: str,
        dataset_path: Optional[str] = None,
    ) -> CompressorMetadata:
        """Compress a model using the provided compression information.

        Args:
            compression (CompressionInfo): The information about the compression.
            output_dir (str): The local path to save the compressed model.
            dataset_path (str, optional): The path of the dataset used for nuclear norm compression method. Default is None.

        Raises:
            e: If an error occurs while compressing the model.

        Returns:
            CompressorMetadata: Compress metadata.
        """

        self.token_handler.validate_token()

        try:
            logger.info("Compressing model...")

            model_info = self.get_model(compression.input_model_id)

            output_dir = FileHandler.create_unique_folder(folder_path=output_dir)
            metadata = self.create_metadata(folder_path=output_dir)
            default_model_path, extension = FileHandler.get_path_and_extension(
                folder_path=output_dir, framework=model_info.detail.framework
            )

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

            for available_layers in compression.available_layers:
                if available_layers.values:
                    available_layers.use = True

            if dataset_path and compression.compression_method == CompressionMethod.PR_NN:
                self.upload_dataset(
                    compression_id=create_compression_response.data.compression_id, dataset_path=dataset_path
                )

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

            compressed_model_info = self.get_model(model_id=compression_info.input_model_id)
            available_options = self._get_available_options(compressed_model_info, default_model_path)

            logger.info(f"Compress model successfully. Compressed Model ID: {compression_info.input_model_id}")

            self.print_remaining_credit()

            if compressed_model_info.detail.framework in [Framework.PYTORCH, Framework.ONNX]:
                metadata.update_compressed_onnx_model_path(
                    compressed_onnx_model_path=default_model_path.with_suffix(".onnx").as_posix()
                )
            metadata.update_compressed_model_path(
                compressed_model_path=default_model_path.with_suffix(extension).as_posix()
            )
            metadata.update_model_info(framework=model_info.detail.framework, input_shapes=model_info.detail.input_layers)
            metadata.update_compression_info(
                method=compression.compression_method,
                options=compression.options,
                layers=compression.available_layers,
            )
            metadata.update_results(model=model_info, compressed_model=compressed_model_info)
            metadata.update_status(status=Status.COMPLETED)
            metadata.update_available_options(available_options)

            MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_dir)

            return metadata

        except Exception as e:
            logger.error(f"Compress model failed. Error: {e}")
            metadata.update_status(status=Status.ERROR)
            MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_dir)
            raise e

        except KeyboardInterrupt:
            logger.error("Compress model stopped.")
            metadata.update_status(status=Status.STOPPED)
            MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_dir)

    def initialize_metadata(self, input_model_path, metadata, framework, input_shapes):
        if (Path(input_model_path).parent / "metadata.json").exists():
            trained_data = FileHandler.load_json(Path(input_model_path).parent / "metadata.json")
            metadata.update_model_info_for_trainer(
                task=trained_data["model_info"]["task"],
                model=trained_data["model_info"]["model"],
                dataset=trained_data["model_info"]["dataset"],
            )
            metadata.update_training_info(
                epochs=trained_data["training_info"]["epochs"],
                batch_size=trained_data["training_info"]["batch_size"],
                learning_rate=trained_data["training_info"]["learning_rate"],
                optimizer=trained_data["training_info"]["optimizer"],
            )
            metadata.update_is_retrainable(is_retrainable=True)

        metadata.update_input_model_path(input_model_path=input_model_path)
        metadata.update_model_info(framework=framework, input_shapes=input_shapes)

        return metadata

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
    ) -> CompressorMetadata:
        """Compress a recommendation-based model using the given compression and recommendation methods.

        Args:
            compression_method (CompressionMethod): The selected compression method.
            recommendation_method (RecommendationMethod): The selected recommendation method.
            recommendation_ratio (float): The compression ratio recommended by the recommendation method.
            input_model_path (str): The file path where the model is located.
            output_dir (str): The local path to save the compressed model.
            input_shapes (List[Dict[str, int]]): Input shapes of the model.
            framework (Framework, optional): The framework of the model.
            options(Options, optional): The options for pruning method.
            dataset_path (str, optional): The path of the dataset used for nuclear norm compression method. Default is None.

        Raises:
            e: If an error occurs while performing recommendation compression.

        Returns:
            CompressorMetadata: Compress metadata.
        """

        self.token_handler.validate_token()

        try:
            logger.info("Compressing recommendation-based model...")

            output_dir = FileHandler.create_unique_folder(folder_path=output_dir)
            metadata = self.create_metadata(folder_path=output_dir)
            metadata = self.initialize_metadata(input_model_path, metadata, framework, input_shapes)
            metadata.compression_info.method = compression_method
            metadata.compression_info.ratio = recommendation_ratio
            MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_dir)

            default_model_path, extension = FileHandler.get_path_and_extension(
                folder_path=output_dir, framework=framework
            )

            self.check_credit_balance(service_credit=ServiceCredit.ADVANCED_COMPRESSION)

            model_info = self.upload_model(
                framework=framework,
                input_model_path=input_model_path,
                input_shapes=input_shapes,
            )

            create_compression_request = RequestCreateCompression(
                ai_model_id=model_info.ai_model_id,
                compression_method=compression_method,
                options=options,
            )
            create_compression_response = compressor_client_v2.create_compression(
                request_data=create_compression_request,
                access_token=self.token_handler.tokens.access_token,
                verify_ssl=self.token_handler.verify_ssl
            )

            if dataset_path and compression_method == CompressionMethod.PR_NN:
                self.upload_dataset(
                    compression_id=create_compression_response.data.compression_id, dataset_path=dataset_path
                )

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

            compressed_model_info = self.get_model(model_id=compression_info.input_model_id)
            available_options = self._get_available_options(compressed_model_info, default_model_path)

            logger.info(f"Recommendation compression successfully. Compressed Model ID: {compression_info.input_model_id}")

            self.print_remaining_credit()

            if compressed_model_info.detail.framework in [Framework.PYTORCH, Framework.ONNX]:
                metadata.update_compressed_onnx_model_path(
                    compressed_onnx_model_path=default_model_path.with_suffix(".onnx").as_posix()
                )
            metadata.update_compressed_model_path(
                compressed_model_path=default_model_path.with_suffix(extension).as_posix()
            )
            metadata.compression_info.layers = compression_info.available_layers
            metadata.compression_info.options = options
            metadata.update_results(model=model_info, compressed_model=compressed_model_info)
            metadata.update_status(status=Status.COMPLETED)
            metadata.update_available_options(available_options)

            MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_dir)

            return metadata

        except Exception as e:
            logger.error(f"Recommendation compression failed. Error: {e}")
            metadata.update_status(status=Status.ERROR)
            MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_dir)
            raise e

        except KeyboardInterrupt:
            logger.error("Recommendation compression stopped.")
            metadata.update_status(status=Status.STOPPED)
            MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_dir)

    def automatic_compression(
        self,
        input_model_path: str,
        output_dir: str,
        input_shapes: List[Dict[str, int]],
        framework: Framework = Framework.PYTORCH,
        compression_ratio: float = 0.5,
    ) -> CompressorMetadata:
        """Compress a model automatically based on the given compression ratio.

        Args:
            input_model_path (str): The file path where the model is located.
            output_dir (str): The local path to save the compressed model.
            input_shapes (List[Dict[str, int]]): Input shapes of the model.
            framework (Framework, optional): The framework of the model.
            compression_ratio (float, optional): The compression ratio for automatic compression. Defaults to 0.5.

        Raises:
            e: If an error occurs while performing automatic compression.

        Returns:
            CompressorMetadata: Compress metadata.
        """
        self.token_handler.validate_token()

        try:
            logger.info("Compressing automatic-based model...")

            output_dir = FileHandler.create_unique_folder(folder_path=output_dir)
            metadata = self.create_metadata(folder_path=output_dir)
            metadata = self.initialize_metadata(input_model_path, metadata, framework, input_shapes)
            metadata.compression_info.method = CompressionMethod.PR_L2
            metadata.compression_info.ratio = compression_ratio
            MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_dir)

            default_model_path, extension = FileHandler.get_path_and_extension(
                folder_path=output_dir, framework=framework
            )

            self.check_credit_balance(service_credit=ServiceCredit.AUTOMATIC_COMPRESSION)

            model_info = self.upload_model(framework=framework, input_model_path=input_model_path, input_shapes=input_shapes)

            logger.info("Compressing model...")
            automatic_compression_request = RequestAutomaticCompressionParams(compression_ratio=compression_ratio)
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

            compressed_model_info = self.get_model(model_id=compression_info.input_model_id)
            available_options = self._get_available_options(compressed_model_info, default_model_path)

            logger.info(f"Automatic compression successfully. Compressed Model ID: {compression_info.input_model_id}")

            self.print_remaining_credit()

            if compressed_model_info.detail.framework in [Framework.PYTORCH, Framework.ONNX]:
                metadata.update_compressed_onnx_model_path(default_model_path.with_suffix(".onnx").as_posix())
            metadata.compression_info.layers = compression_info.available_layers
            metadata.compression_info.options = compression_info.options
            metadata.update_compressed_model_path(default_model_path.with_suffix(extension).as_posix())
            metadata.update_results(model=model_info, compressed_model=compressed_model_info)
            metadata.update_status(status=Status.COMPLETED)
            metadata.update_available_options(available_options)

            MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_dir)

            return metadata

        except Exception as e:
            logger.error(f"Automatic compression failed. Error: {e}")
            metadata.update_status(status=Status.ERROR)
            MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_dir)
            raise e

        except KeyboardInterrupt:
            logger.error("Automatic compression stopped.")
            metadata.update_status(status=Status.STOPPED)
            MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_dir)
