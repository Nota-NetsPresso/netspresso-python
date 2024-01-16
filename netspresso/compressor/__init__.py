import sys
from pathlib import Path
from typing import Dict, List, Union
from urllib import request

from loguru import logger

from netspresso.clients.auth import BaseClient, validate_token
from netspresso.clients.compressor import ModelCompressorAPIClient
from netspresso.clients.compressor.enums import (
    CompressionMethod,
    Extension,
    Framework,
    GroupPolicy,
    LayerNorm,
    OriginFrom,
    Policy,
    RecommendationMethod,
    Task,
)
from netspresso.clients.compressor.schemas.compression import (
    AutoCompressionRequest,
    AvailableLayer,
    CompressionRequest,
    CreateCompressionRequest,
    GetAvailableLayersRequest,
    Options,
    RecommendationRequest,
    UploadDatasetRequest,
)
from netspresso.clients.compressor.schemas.model import UploadModelRequest
from netspresso.compressor.core.compression import CompressionInfo
from netspresso.compressor.core.model import (
    CompressedModel,
    Model,
    ModelCollection,
    ModelFactory,
)
from netspresso.enums import ServiceCredit, TaskType, Status

from ..utils import FileManager, check_credit_balance
from .utils.onnx import export_onnx
from ..utils.metadata.manager import MetadataManager


class ModelCompressor(BaseClient):
    def __init__(self, email=None, password=None, user_session=None):
        """Initialize the Model Compressor.

        Args:
            email (str): The email address for a user account.
            password (str): The password for a user account.
            user_session (SessionClient): The SessionClient object.

        Available constructors:
            ModelCompressor(email='USER_EMAIL',password='PASSWORD')
            ModelCompressor(user_session=SessionClient(email='USER_EMAIL',password='PASSWORD')
        """
        super().__init__(email=email, password=password, user_session=user_session)
        self.client = ModelCompressorAPIClient()
        self.model_factory = ModelFactory()

    @validate_token
    def upload_model(
        self,
        model_name: str,
        task: Task,
        framework: Framework,
        file_path: str,
        input_shapes: List[Dict[str, int]] = [],
    ) -> Model:
        """Upload a model for compression.

        Args:
            model_name (str): The name of the model.
            task (Task): The task of the model.
            framework (Framework): The framework of the model.
            file_path (str): The file path where the model is located.
            input_shapes (List[Dict[str, int]], optional): Input shapes of the model. Defaults to [].

        Raises:
            e: If an error occurs while uploading the model.

        Returns:
            Model: Uploaded model object.
        """

        try:
            logger.info("Uploading Model...")
            data = UploadModelRequest(
                model_name=model_name,
                task=task,
                framework=framework,
                file_path=file_path,
                input_layers=input_shapes,
            )
            model_info = self.client.upload_model(
                data=data, access_token=self.user_session.access_token
            )
            model = self.model_factory.create_model(model_info=model_info)

            logger.info(f"Upload model successfully. Model ID: {model.model_id}")

            return model

        except Exception as e:
            logger.error(f"Upload model failed. Error: {e}")
            raise e

    @validate_token
    def get_models(self) -> List[ModelCollection]:
        """Get the list of uploaded & compressed models.

        Raises:
            e: If an error occurs while getting the model list.

        Returns:
            List[ModelCollection]: The list of uploaded & compressed models.
        """

        try:
            logger.info("Getting model list...")
            models = []
            parent_models = self.client.get_parent_models(
                is_simple=True, access_token=self.user_session.access_token
            )

            for parent_model_info in parent_models:
                if parent_model_info.origin_from == "custom":
                    children_models = self.client.get_children_models(
                        model_id=parent_model_info.model_id,
                        access_token=self.user_session.access_token,
                    )
                    model_collection = self.model_factory.create_model_collection(
                        model_info=parent_model_info, children_models=children_models
                    )
                    models.append(model_collection)
            logger.info("Get model list successfully.")

            return models

        except Exception as e:
            logger.error(f"Get model list failed. Error: {e}")
            raise e

    @validate_token
    def get_uploaded_models(self) -> List[Model]:
        """Get the list of uploaded models.

        Raises:
            e: If an error occurs while getting the uploaded model list.

        Returns:
            List[Model]: The list of uploaded models.
        """

        try:
            logger.info("Getting uploaded model list...")
            parent_models = self.client.get_parent_models(
                is_simple=True, access_token=self.user_session.access_token
            )
            uploaded_models = [
                self.model_factory.create_model(model_info=parent_model_info)
                for parent_model_info in parent_models
                if parent_model_info.origin_from == "custom"
            ]
            logger.info("Get uploaded model list successfully.")

            return uploaded_models

        except Exception as e:
            logger.error(f"Get uploaded model list failed. Error: {e}")
            raise e

    @validate_token
    def get_compressed_models(self, model_id: str) -> List[CompressedModel]:
        """Get the list of compressed models for a given model ID.

        Args:
            model_id (str): The ID of the model.

        Raises:
            e: If an error occurs while getting the compressed model list.

        Returns:
            List[CompressedModel]: The list of compressed models for a given model ID.
        """

        try:
            logger.info("Getting compressed model list...")
            children_models = self.client.get_children_models(
                model_id=model_id, access_token=self.user_session.access_token
            )
            compressed_models = [
                self.model_factory.create_compressed_model(
                    model_info=children_model_info
                )
                for children_model_info in children_models
            ]
            logger.info("Get compressed model list successfully.")

            return compressed_models

        except Exception as e:
            logger.error(f"Get compressed model list failed. Error: {e}")
            raise e

    @validate_token
    def get_model(self, model_id: str) -> Union[Model, CompressedModel]:
        """Get the model for a given model ID.

        Args:
            model_id (str): The ID of the model.

        Raises:
            e: If an error occurs while getting the model.

        Returns:
            Union[Model, CompressedModel]: The retrieved model. If the model is compressed,
            `CompressedModel` will be returned. Otherwise, `Model` will be returned.
        """

        try:
            logger.info("Getting model...")
            model_info = self.client.get_model_info(
                model_id=model_id, access_token=self.user_session.access_token
            )
            if model_info.status.is_compressed:
                model = self.model_factory.create_compressed_model(
                    model_info=model_info
                )
            else:
                model = self.model_factory.create_model(model_info=model_info)
            logger.info("Get model successfully.")

            return model

        except Exception as e:
            logger.error(f"Get model failed. Error: {e}")
            raise e

    @validate_token
    def download_model(self, model_id: str, local_path: str) -> None:
        """Download the model for a given model ID to the local path.

        Args:
            model_id (str): The ID of the model.
            local_path (str): The local path to save the downloaded model.

        Raises:
            e: If an error occurs while downloading the model.
        """

        try:
            logger.info("Downloading model...")
            download_link = self.client.get_download_model_link(
                model_id=model_id, access_token=self.user_session.access_token
            )
            request.urlretrieve(download_link.url, local_path)
            logger.info(f"Model downloaded at {Path(local_path)}")

        except Exception as e:
            logger.error(f"Download model failed. Error: {e}")
            raise e

    @validate_token
    def delete_model(self, model_id: str, recursive: bool = False) -> None:
        """Delete the model for a given model ID.

        Args:
            model_id (str): The ID of the model.
            recursive (bool, optional): Whether to also delete the compressed model for that model. Defaults to False.

        Raises:
            e: If an error occurs while deleting the model.
        """

        try:
            logger.info("Deleting model...")
            children_models = self.client.get_children_models(
                model_id=model_id, access_token=self.user_session.access_token
            )
            if len(children_models) != 0:
                if not recursive:
                    logger.warning(
                        "Deleting the model will also delete its compressed models. To proceed with the deletion, set the `recursive` parameter to True."
                    )
                else:
                    logger.info(
                        "The compressed model for that model will also be deleted."
                    )
                    self.client.delete_model(
                        model_id=model_id, access_token=self.user_session.access_token
                    )
                    logger.info("Delete model successfully.")
            else:
                logger.info("The model will be deleted.")
                self.client.delete_model(
                    model_id=model_id, access_token=self.user_session.access_token
                )
                logger.info("Delete model successfully.")

        except Exception as e:
            logger.error(f"Delete model failed. Error: {e}")
            raise e

    @validate_token
    def select_compression_method(
        self,
        model_id: str,
        compression_method: CompressionMethod,
        options: Options = Options(),
    ) -> CompressionInfo:
        """Select a compression method for a model.

        Args:
            model_id (str): The ID of the model.
            compression_method (CompressionMethod): The selected compression method.
            options(Options, optional): The options for pruning method.

        Raises:
            e: If an error occurs while selecting the compression method.

        Returns:
            CompressionInfo: The compression information for the selected compression method.
        """
        try:
            model = self.get_model(model_id)

            logger.info("Selecting compression method...")
            if (
                model.framework == Framework.PYTORCH
                and compression_method == CompressionMethod.PR_NN
            ):
                raise Exception(
                    "The Nuclear Norm is only supported in the TensorFlow-Keras framework."
                )

            data = GetAvailableLayersRequest(
                model_id=model.model_id,
                compression_method=compression_method,
                options=options,
            )
            response = self.client.get_available_layers(
                data=data, access_token=self.user_session.access_token
            )
            compression_info = CompressionInfo(
                original_model_id=model.model_id,
                compression_method=compression_method,
                options=options.dict(),
            )
            compression_info.set_available_layers(response.available_layers)
            logger.info("Select compression method successfully.")

            return compression_info

        except Exception as e:
            logger.error(f"Select compression method failed. Error: {e}")
            raise e

    @validate_token
    def get_compression(self, compression_id: str) -> CompressionInfo:
        """Get information about a compression.

        Args:
            compression_id (str): The ID of the compression.

        Raises:
            e: If an error occurs while getting the compression information.

        Returns:
            CompressionInfo: The information about the compression.
        """
        try:
            logger.info("Getting compression...")
            _compression_info = self.client.get_compression_info(
                compression_id=compression_id,
                access_token=self.user_session.access_token,
            )
            compression_info = CompressionInfo(
                compressed_model_id=_compression_info.new_model_id,
                compression_id=_compression_info.compression_id,
                compression_method=_compression_info.compression_method,
            )
            compression_info.set_available_layers(_compression_info.available_layers)
            logger.info("Get compression successfully.")

            return compression_info

        except Exception as e:
            logger.error(f"Get compression failed. Error: {e}")
            raise e

    @validate_token
    def __upload_dataset(self, model_id: str, dataset_path: str) -> None:
        """Upload a dataset for nuclear norm compression method.

        Args:
            model_id (str): The ID of the model.
            dataset_path (str): The file path where the dataset is located.

        Raises:
            e: If an error occurs while uploading the dataset.
        """
        try:
            logger.info(f"Uploading dataset...")
            data = UploadDatasetRequest(model_id=model_id, file_path=dataset_path)
            self.client.upload_dataset(
                data=data, access_token=self.user_session.access_token
            )
            logger.info(f"Upload dataset successfully.")

        except Exception as e:
            logger.error(f"Upload dataset failed. Error: {e}")
            raise e

    @validate_token
    def compress_model(
        self,
        compression: CompressionInfo,
        model_name: str,
        output_path: str,
        dataset_path: str = None,
    ) -> CompressedModel:
        """Compress a model using the provided compression information.

        Args:
            compression (CompressionInfo): The information about the compression.
            model_name (str): The name of the compressed model.
            output_path (str): The local path to save the compressed model.
            dataset_path (str, optional): The path of the dataset used for nuclear norm compression method. Default is None.

        Raises:
            e: If an error occurs while compressing the model.

        Returns:
            CompressedModel: The compressed model.
        """
        try:
            logger.info("Compressing model...")

            model_info = self.get_model(compression.original_model_id)

            default_model_path, extension = FileManager.prepare_model_path(
                folder_path=output_path, framework=model_info.framework
            )
            metadata = MetadataManager.init_metadata(folder_path=output_path, task_type=TaskType.COMPRESS)

            current_credit = self.user_session.get_credit()
            check_credit_balance(
                user_credit=current_credit,
                service_credit=ServiceCredit.ADVANCED_COMPRESSION,
            )

            data = CreateCompressionRequest(
                model_id=compression.original_model_id,
                model_name=model_name,
                compression_method=compression.compression_method,
                options=compression.options,
            )
            compression_info = self.client.create_compression(
                data=data, access_token=self.user_session.access_token
            )

            if (
                dataset_path
                and compression.compression_method == CompressionMethod.PR_NN
            ):
                self.__upload_dataset(
                    model_id=compression.original_model_id, dataset_path=dataset_path
                )

            for available_layers in compression.available_layers:
                if available_layers.values != [""]:
                    available_layers.use = True

            all_layers_false = all(
                available_layer.values == [""]
                for available_layer in compression.available_layers
            )
            if all_layers_false:
                raise Exception(
                    f"The available_layer.values all empty. please put in the available_layer.values to compress."
                )

            available_layers = [
                AvailableLayer(
                    name=layer.name,
                    values=layer.values,
                    channels=layer.channels,
                    use=layer.use,
                )
                for layer in compression.available_layers
            ]

            data = CompressionRequest(
                compression_id=compression_info.compression_id,
                compression_method=compression.compression_method,
                layers=available_layers,
                compressed_model_id=compression_info.new_model_id,
                options=compression.options,
            )
            self.client.compress_model(
                data=data, access_token=self.user_session.access_token
            )

            self.download_model(
                model_id=compression_info.new_model_id,
                local_path=default_model_path.with_suffix(extension),
            )
            compressed_model = self.get_model(model_id=compression_info.new_model_id)

            if compressed_model.framework in [Framework.PYTORCH, Framework.ONNX]:
                export_onnx(default_model_path, compressed_model.input_shapes)

            logger.info(
                f"Compress model successfully. Compressed Model ID: {compressed_model.model_id}"
            )
            remaining_credit = self.user_session.get_credit()
            logger.info(
                f"{ServiceCredit.ADVANCED_COMPRESSION} credits have been consumed. Remaining Credit: {remaining_credit}"
            )

            metadata.update_model_info(task=model_info.task, framework=model_info.framework, input_shapes=model_info.input_shapes)
            metadata.update_compression_info(
                method=compression.compression_method,
                options=compression.options,
                layers=compression,
            )
            metadata.update_results(model=model_info, compressed_model=compressed_model)
            metadata.update_status(status=Status.COMPLETED)
            MetadataManager.save_json(data=metadata.asdict(), folder_path=output_path)

            return compressed_model

        except Exception as e:
            logger.error(f"Compress model failed. Error: {e}")
            metadata.update_status(status=Status.ERROR)
            MetadataManager.save_json(data=metadata.asdict(), folder_path=output_path)
            raise e

        except KeyboardInterrupt:
            metadata.update_status(status=Status.STOPPED)
            MetadataManager.save_json(data=metadata.asdict(), folder_path=output_path)

    @validate_token
    def recommendation_compression(
        self,
        model_name: str,
        compression_method: CompressionMethod,
        recommendation_method: RecommendationMethod,
        recommendation_ratio: float,
        input_path: str,
        output_path: str,
        task: Task,
        framework: Framework,
        input_shapes: List[Dict[str, int]],
        options: Options = Options(),
        dataset_path: str = None,
    ) -> CompressedModel:
        """Compress a recommendation-based model using the given compression and recommendation methods.

        Args:
            model_name (str): The name of the model.
            compression_method (CompressionMethod): The selected compression method.
            recommendation_method (RecommendationMethod): The selected recommendation method.
            recommendation_ratio (float): The compression ratio recommended by the recommendation method.
            input_path (str): The file path where the model is located.
            output_path (str): The local path to save the compressed model.
            task (Task): The task of the model.
            framework (Framework): The framework of the model.
            input_shapes (List[Dict[str, int]], optional): Input shapes of the model. Defaults to [].
            options(Options, optional): The options for pruning method.
            dataset_path (str, optional): The path of the dataset used for nuclear norm compression method. Default is None.

        Raises:
            e: If an error occurs while performing recommendation compression.

        Returns:
            CompressedModel: The compressed model.
        """

        try:
            logger.info("Compressing recommendation-based model...")

            default_model_path, extension = FileManager.prepare_model_path(
                folder_path=output_path, framework=framework
            )
            metadata = MetadataManager.init_metadata(folder_path=output_path, task_type=TaskType.COMPRESS)

            current_credit = self.user_session.get_credit()
            check_credit_balance(
                user_credit=current_credit,
                service_credit=ServiceCredit.ADVANCED_COMPRESSION,
            )

            if (
                framework == Framework.PYTORCH
                and compression_method == CompressionMethod.PR_NN
            ):
                raise Exception(
                    "The Nuclear Norm is only supported in the TensorFlow-Keras framework."
                )

            if compression_method in [CompressionMethod.PR_ID, CompressionMethod.FD_CP]:
                raise Exception(
                    f"The {compression_method} compression method you choose doesn't provide a recommendation."
                )

            if (
                compression_method
                in [
                    CompressionMethod.PR_L2,
                    CompressionMethod.PR_GM,
                    CompressionMethod.PR_NN,
                ]
                and recommendation_method != RecommendationMethod.SLAMP
            ):
                raise Exception(
                    f"The {compression_method} compression method is only available the SLAMP recommendation method."
                )

            if (
                compression_method
                in [CompressionMethod.FD_TK, CompressionMethod.FD_SVD]
                and recommendation_method != RecommendationMethod.VBMF
            ):
                raise Exception(
                    f"The {compression_method} compression method is only available the VBMF recommendation method."
                )

            model = self.upload_model(
                model_name=model_name,
                task=task,
                framework=framework,
                file_path=input_path,
                input_shapes=input_shapes,
            )

            data = CreateCompressionRequest(
                model_id=model.model_id,
                model_name=model_name,
                compression_method=compression_method,
                options=options.dict(),
            )
            compression_info = self.client.create_compression(
                data=data, access_token=self.user_session.access_token
            )

            if dataset_path and compression_method == CompressionMethod.PR_NN:
                self.__upload_dataset(
                    model_id=model.model_id, dataset_path=dataset_path
                )

            data = RecommendationRequest(
                model_id=model.model_id,
                compression_id=compression_info.compression_id,
                recommendation_method=recommendation_method,
                recommendation_ratio=recommendation_ratio,
                options=options.dict(),
            )
            logger.info("Compressing model...")
            recommendation_result = self.client.get_recommendation(
                data=data, access_token=self.user_session.access_token
            )

            for recommended_layer in recommendation_result.recommended_layers:
                for available_layer in compression_info.available_layers:
                    # Find the matching available_layer by name
                    if available_layer.name == recommended_layer.name:
                        available_layer.use = True
                        available_layer.values = recommended_layer.values

            data = CompressionRequest(
                compression_id=compression_info.compression_id,
                compression_method=compression_method,
                layers=compression_info.available_layers,
                compressed_model_id=compression_info.new_model_id,
                options=options.dict(),
            )
            self.client.compress_model(
                data=data, access_token=self.user_session.access_token
            )

            self.download_model(
                model_id=compression_info.new_model_id,
                local_path=default_model_path.with_suffix(extension),
            )
            compressed_model = self.get_model(model_id=compression_info.new_model_id)

            if compressed_model.framework in [Framework.PYTORCH, Framework.ONNX]:
                export_onnx(default_model_path, compressed_model.input_shapes)

            logger.info(
                f"Recommendation compression successfully. Compressed Model ID: {compressed_model.model_id}"
            )
            remaining_credit = self.user_session.get_credit()
            logger.info(
                f"{ServiceCredit.ADVANCED_COMPRESSION} credits have been consumed. Remaining Credit: {remaining_credit}"
            )

            _compression_info = self.get_compression(compression_info.compression_id)
            metadata.update_model_info(task=task, framework=framework, input_shapes=input_shapes)
            metadata.update_compression_info(
                method=_compression_info.compression_method,
                ratio=recommendation_ratio,
                options=_compression_info.options,
                layers=_compression_info.available_layers,
            )
            metadata.update_results(model=model, compressed_model=compressed_model)
            metadata.update_status(status=Status.COMPLETED)
            MetadataManager.save_json(data=metadata.asdict(), folder_path=output_path)

            return compressed_model

        except Exception as e:
            logger.error(f"Recommendation compression failed. Error: {e}")
            metadata.update_status(status=Status.ERROR)
            MetadataManager.save_json(data=metadata.asdict(), folder_path=output_path)
            raise e

        except KeyboardInterrupt:
            metadata.update_status(status=Status.STOPPED)
            MetadataManager.save_json(data=metadata.asdict(), folder_path=output_path)

    @validate_token
    def automatic_compression(
        self,
        model_name: str,
        task: Task,
        framework: Framework,
        input_shapes: List[Dict[str, int]],
        input_path: str,
        output_path: str,
        compression_ratio: float = 0.5,
    ) -> CompressedModel:
        """Compress a model automatically based on the given compression ratio.

        Args:
            model_name (str): The name of the model.
            task (Task): The task of the model.
            framework (Framework): The framework of the model.
            input_shapes (List[Dict[str, int]], optional): Input shapes of the model. Defaults to [].
            input_path (str): The file path where the model is located.
            output_path (str): The local path to save the compressed model.
            compression_ratio (float): The compression ratio for automatic compression. Defaults to 0.5.

        Raises:
            e: If an error occurs while performing automatic compression.

        Returns:
            CompressedModel: The compressed model.
        """

        try:
            logger.info("Compressing automatic-based model...")

            default_model_path, extension = FileManager.prepare_model_path(
                folder_path=output_path, framework=framework
            )
            metadata = MetadataManager.init_metadata(folder_path=output_path, task_type=TaskType.COMPRESS)

            current_credit = self.user_session.get_credit()
            check_credit_balance(
                user_credit=current_credit,
                service_credit=ServiceCredit.AUTOMATIC_COMPRESSION,
            )

            model = self.upload_model(
                model_name=model_name,
                task=task,
                framework=framework,
                file_path=input_path,
                input_shapes=input_shapes,
            )

            compressed_model_name = f"{model_name}_automatic_{compression_ratio}"
            data = AutoCompressionRequest(
                model_id=model.model_id,
                model_name=compressed_model_name,
                recommendation_ratio=compression_ratio,
                save_path=output_path,
            )
            logger.info("Compressing model...")
            model_info = self.client.auto_compression(
                data=data, access_token=self.user_session.access_token
            )
            compression_info = self.get_compression(model_info.original_compression_id)

            self.download_model(
                model_id=model_info.model_id,
                local_path=default_model_path.with_suffix(extension),
            )
            compressed_model = self.model_factory.create_compressed_model(
                model_info=model_info
            )  # TODO: delete

            if compressed_model.framework in [Framework.PYTORCH, Framework.ONNX]:
                export_onnx(default_model_path, compressed_model.input_shapes)

            logger.info(
                f"Automatic compression successfully. Compressed Model ID: {compressed_model.model_id}"
            )
            remaining_credit = self.user_session.get_credit()
            logger.info(
                f"{ServiceCredit.AUTOMATIC_COMPRESSION} credits have been consumed. Remaining Credit: {remaining_credit}"
            )

            metadata.update_model_info(task=task, framework=framework, input_shapes=input_shapes)
            metadata.update_compression_info(
                method=compression_info.compression_method,
                ratio=compression_ratio,
                options=compression_info.options,
                layers=compression_info.available_layers,
            )
            metadata.update_results(model=model, compressed_model=compressed_model)
            metadata.update_status(status=Status.COMPLETED)
            MetadataManager.save_json(data=metadata.asdict(), folder_path=output_path)

            return compressed_model

        except Exception as e:
            logger.error(f"Automatic compression failed. Error: {e}")
            metadata.update_status(status=Status.ERROR)
            MetadataManager.save_json(data=metadata.asdict(), folder_path=output_path)
            raise e

        except KeyboardInterrupt:
            metadata.update_status(status=Status.STOPPED)
            MetadataManager.save_json(data=metadata.asdict(), folder_path=output_path)
