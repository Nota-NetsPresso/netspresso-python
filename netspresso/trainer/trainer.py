from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from omegaconf import OmegaConf

from netspresso.clients.auth import TokenHandler
from netspresso.clients.launcher import launcher_client_v2
from netspresso.enums import Framework, Status, Task
from netspresso.metadata.common import InputShape
from netspresso.metadata.trainer import TrainerMetadata
from netspresso.trainer.augmentations import AUGMENTATION_CONFIG_TYPE, AugmentationConfig, Transform
from netspresso.trainer.data import DATA_CONFIG_TYPE, ImageLabelPathConfig, PathConfig
from netspresso.trainer.models import (
    CLASSIFICATION_MODELS,
    DETECTION_MODELS,
    SEGMENTATION_MODELS,
    CheckpointConfig,
    ModelConfig,
)
from netspresso.trainer.trainer_configs import TrainerConfigs
from netspresso.trainer.training import TRAINING_CONFIG_TYPE, EnvironmentConfig, LoggingConfig, ScheduleConfig
from netspresso.utils import FileHandler
from netspresso.utils.metadata import MetadataHandler


class Trainer:
    def __init__(
        self, token_handler: TokenHandler, task: Optional[Union[str, Task]] = None, yaml_path: Optional[str] = None
    ) -> None:
        """Initialize the Trainer.

        Args:
            task (Union[str, Task]], optional): The type of task (classification, detection, segmentation). Either 'task' or 'yaml_path' must be provided, but not both.
            yaml_path (str, optional): Path to the YAML configuration file. Either 'task' or 'yaml_path' must be provided, but not both.
        """

        self.token_handler = token_handler

        if (task is not None) == (yaml_path is not None):
            raise ValueError("Either 'task' or 'yaml_path' must be provided, but not both.")

        if task is not None:
            self._initialize_from_task(task)
        elif yaml_path is not None:
            self._initialize_from_yaml(yaml_path)

    def _initialize_from_task(self, task: Union[str, Task]) -> None:
        """Initialize the Trainer object based on the provided task.

        Args:
            task (Union[str, Task]): The task for which the Trainer is initialized.
        """

        self.task = self._validate_task(task)
        self.available_models = list(self._get_available_models().keys())
        self.data = None
        self.model = None
        self.training = TRAINING_CONFIG_TYPE[self.task]()
        self.augmentation = AUGMENTATION_CONFIG_TYPE[self.task]()
        self.logging = LoggingConfig()
        self.environment = EnvironmentConfig()

    def _initialize_from_yaml(self, yaml_path: str) -> None:
        """Initialize the Trainer object based on the configuration provided in a YAML file.

        Args:
            yaml_path (str): The path to the YAML file containing the configuration.
        """

        hparams = OmegaConf.load(yaml_path)
        hparams["model"].pop("single_task_model")

        self.img_size = hparams["augmentation"]["img_size"]
        self.task = hparams["data"]["task"]
        self.available_models = list(self._get_available_models().keys())

        self.data = DATA_CONFIG_TYPE[self.task](**hparams["data"])
        self.model = ModelConfig(**hparams["model"])
        self.training = ScheduleConfig(**hparams["training"])
        self.augmentation = AugmentationConfig(**hparams["augmentation"])
        self.logging = LoggingConfig(**hparams["logging"])
        self.environment = EnvironmentConfig(**hparams["environment"])

    def _validate_task(self, task: Union[str, Task]):
        """Validate the provided task.

        Args:
            task (Union[str, Task]): The task to be validated.

        Raises:
            ValueError: If the provided task is not supported.

        Returns:
            Task: The validated task.
        """

        available_tasks = [task.value for task in Task]
        if task not in available_tasks:
            raise ValueError(f"The task supports {available_tasks}. The entered task is {task}.")
        return task

    def _validate_config(self):
        """Validate the configuration setup.

        Raises:
            ValueError: Raised if the dataset is not set. Use `set_dataset_config` or `set_dataset_config_with_yaml` to set the dataset configuration.
            ValueError: Raised if the model is not set. Use `set_model_config` or `set_model_config_with_yaml` to set the model configuration.
        """

        if self.data is None:
            raise ValueError(
                "The dataset is not set. Use `set_dataset_config` or `set_dataset_config_with_yaml` to set the dataset configuration."
            )
        if self.model is None:
            raise ValueError(
                "The model is not set. Use `set_model_config` or `set_model_config_with_yaml` to set the model configuration."
            )

    def _get_available_models(self) -> Dict[str, Any]:
        """Get available models based on the current task.

        Returns:
            Dict[str, Any]: A dictionary mapping model types to available models.
        """

        available_models = {
            "classification": CLASSIFICATION_MODELS,
            "detection": DETECTION_MODELS,
            "segmentation": SEGMENTATION_MODELS,
        }[self.task]

        return available_models

    def set_dataset_config(
        self,
        name: str,
        root_path: str,
        train_image: str = "images/train",
        train_label: str = "labels/train",
        valid_image: str = "images/valid",
        valid_label: str = "labels/valid",
        id_mapping: Optional[Union[List[str], Dict[str, str], str]] = None,
    ):
        """Set the dataset configuration for the Trainer.

        Args:
            name (str): The name of dataset.
            root_path (str): Root directory of dataset.
            train_image (str, optional): The directory for training images. Should be relative path to root directory. Defaults to "images/train".
            train_label (str, optional): The directory for training labels. Should be relative path to root directory. Defaults to "labels/train".
            valid_image (str, optional): The directory for validation images. Should be relative path to root directory. Defaults to "images/val".
            valid_label (str, optional): The directory for validation labels. Should be relative path to root directory. Defaults to "labels/val".
            id_mapping (Union[List[str], Dict[str, str]], optional): ID mapping for the dataset. Defaults to None.
        """

        common_config = {
            "name": name,
            "path": PathConfig(
                root=root_path,
                train=ImageLabelPathConfig(image=train_image, label=train_label),
                valid=ImageLabelPathConfig(image=valid_image, label=valid_label),
            ),
            "id_mapping": id_mapping,
        }
        self.data = DATA_CONFIG_TYPE[self.task](**common_config)

    def check_paths_exist(self, base_path):
        paths = [
            "images/train",
            "labels/train",
            "images/valid",
            "labels/valid",
            "id_mapping.json",
        ]

        for relative_path in paths:
            path = Path(base_path) / relative_path
            if not path.exists():
                if path.suffix:
                    raise FileNotFoundError(
                        f"The required file '{relative_path}' does not exist. Please check and make sure it is in the correct location."
                    )
                else:
                    raise FileNotFoundError(
                        f"The required directory '{relative_path}' does not exist. Please check and make sure it is in the correct location."
                    )

    def set_dataset(self, dataset_root_path: str):
        dataset_name = Path(dataset_root_path).name

        self.check_paths_exist(dataset_root_path)
        self.set_dataset_config(
            name=dataset_name,
            root_path=dataset_root_path,
            id_mapping="id_mapping.json",
        )

    def set_model_config(
        self,
        model_name: str,
        img_size: int,
        use_pretrained: bool = True,
        load_head: bool = False,
        path: Optional[str] = None,
        fx_model_path: Optional[str] = None,
        optimizer_path: Optional[str] = None,
    ):
        """Set the model configuration for the Trainer.

        Args:
            model_name (str): Name of the model.
            img_size (int): Image size for the model.
            use_pretrained (bool, optional): Whether to use a pre-trained model. Defaults to True.
            load_head (bool, optional): Whether to load the model head. Defaults to False.
            path (str, optional): Path to the model. Defaults to None.
            fx_model_path (str, optional): Path to the FX model. Defaults to None.
            optimizer_path (str, optional): Path to the optimizer. Defaults to None.

        Raises:
            ValueError: If the specified model is not supported for the current task.
        """

        model = self._get_available_models().get(model_name)
        self.img_size = img_size

        if model is None:
            raise ValueError(
                f"The '{model_name}' model is not supported for the '{self.task}' task. The available models are {self.available_models}."
            )

        self.model = model(
            checkpoint=CheckpointConfig(
                use_pretrained=use_pretrained,
                load_head=load_head,
                path=path,
                fx_model_path=fx_model_path,
                optimizer_path=optimizer_path,
            )
        )

    def set_fx_model(self, fx_model_path: str):
        """Set the FX model path for retraining.

        Args:
            fx_model_path (str): The path to the FX model.

        Raises:
            ValueError: If the model is not set. Please use 'set_model_config' for model setup.
        """

        if not self.model:
            raise ValueError("This function is intended for retraining. Please use 'set_model_config' for model setup.")

        self.model.checkpoint.path = None
        self.model.checkpoint.fx_model_path = fx_model_path

    def set_training_config(
        self,
        optimizer,
        scheduler,
        epochs: int = 3,
        batch_size: int = 8,
    ):
        """Set the training configuration.

        Args:
            optimizer: The configuration of optimizer.
            scheduler: The configuration of learning rate scheduler.
            epochs (int, optional): The total number of epoch for training the model. Defaults to 3.
            batch_size (int, optional): The number of samples in single batch input. Defaults to 8.
        """

        self.training = ScheduleConfig(
            epochs=epochs,
            optimizer=optimizer.asdict(),
            scheduler=scheduler.asdict(),
        )
        self.environment.batch_size = batch_size

    def set_augmentation_config(
        self,
        train_transforms: Optional[List] = None,
        inference_transforms: Optional[List] = None,
    ):
        """Set the augmentation configuration for training.

        Args:
            train_transforms (List, optional): List of transforms for training. Defaults to None.
            inference_transforms (List, optional): List of transforms for inference. Defaults to None.
        """

        self.augmentation = AugmentationConfig(
            train=train_transforms,
            inference=inference_transforms,
        )

    def set_logging_config(
        self,
        project_id: Optional[str] = None,
        output_dir: str = "./outputs",
        tensorboard: bool = True,
        csv: bool = False,
        image: bool = True,
        stdout: bool = True,
        save_optimizer_state: bool = True,
        validation_epoch: int = 10,
        save_checkpoint_epoch: Optional[int] = None,
    ):
        """Set the logging configuration.

        Args:
            project_id (str, optional): Project name to save the experiment. If None, it is set as {task}_{model} (e.g. segmentation_segformer).
            output_dir (str, optional): Root directory for saving the experiment. Defaults to "./outputs".
            tensorboard (bool, optional): Whether to use the tensorboard. Defaults to True.
            csv (bool, optional): Whether to save the result in csv format. Defaults to False.
            image (bool, optional): Whether to save the validation results. It is ignored if the task is classification. Defaults to True.
            stdout (bool, optional): Whether to log the standard output. Defaults to True.
            save_optimizer_state (bool, optional): Whether to save optimizer state with model checkpoint to resume training. Defaults to True.
            validation_epoch (int, optional): Validation frequency in total training process. Defaults to 10.
            save_checkpoint_epoch (int, optional): Checkpoint saving frequency in total training process. Defaults to None.
        """

        self.logging = LoggingConfig(
            project_id=project_id,
            output_dir=output_dir,
            tensorboard=tensorboard,
            csv=csv,
            image=image,
            stdout=stdout,
            save_optimizer_state=save_optimizer_state,
            validation_epoch=validation_epoch,
            save_checkpoint_epoch=save_checkpoint_epoch,
        )

    def set_environment_config(self, seed: int = 1, num_workers: int = 4):
        """Set the environment configuration.

        Args:
            seed (int, optional): Random seed. Defaults to 1.
            num_workers (int, optional): The number of multi-processing workers to be used by the data loader. Defaults to 4.
        """

        self.environment = EnvironmentConfig(seed=seed, num_workers=num_workers)

    def _change_transforms(self, transforms: Transform):
        """Update the 'size' attribute in the given list of transforms with the specified image size.

        Args:
            transforms (List[Transform]): The list of transforms to be updated.

        Returns:
            List[Transform]: The list of transforms with the 'size' attribute updated to the specified image size.
        """

        field_name_to_check = "size"

        if transforms is None:
            return transforms

        for transform in transforms:
            field_type = transform.__annotations__.get(field_name_to_check)

            if field_type == List:
                transform.size = [self.img_size, self.img_size]
            elif field_type == int:
                transform.size = self.img_size

        return transforms

    def _apply_img_size(self):
        """Apply the specified image size to the augmentation configurations.

        This method updates the 'img_size' attribute in the augmentation configurations, including
        'train.transforms', 'train.mix_transforms', and 'inference.transforms'.
        """

        self.augmentation.img_size = self.img_size
        self.augmentation.train = self._change_transforms(self.augmentation.train)
        self.augmentation.inference = self._change_transforms(self.augmentation.inference)

    def _get_available_options(self):
        options_response = launcher_client_v2.converter.read_framework_options(
            access_token=self.token_handler.tokens.access_token,
            framework=Framework.ONNX,
        )

        available_options = options_response.data

        return available_options

    def train(self, gpus: str, project_name: str, output_dir: Optional[str] = "./outputs") -> TrainerMetadata:
        """Train the model with the specified configuration.

        Args:
            gpus (str): GPU ids to use, separated by commas.
            project_name (str): Project name to save the experiment.

        Returns:
            Dict: A dictionary containing information about the training.
        """

        from netspresso_trainer import train_with_yaml

        self._validate_config()
        self._apply_img_size()

        self.logging.output_dir = output_dir
        destination_folder = Path(output_dir) / project_name
        destination_folder = FileHandler.create_unique_folder(folder_path=destination_folder)
        metadata = TrainerMetadata()
        metadata.update_logging_dir(logging_dir=destination_folder)
        metadata.update_model_info(
            task=self.task,
            model=self.model.name,
            dataset=self.data.name,
            input_shapes=[InputShape(batch=1, channel=3, dimension=[self.img_size, self.img_size])],
        )
        metadata.update_training_info(
            epochs=self.training.epochs,
            batch_size=self.environment.batch_size,
            learning_rate=self.training.optimizer["lr"],
            optimizer=self.training.optimizer["name"],
        )
        MetadataHandler.save_json(data=metadata.asdict(), folder_path=destination_folder)
        self.logging.project_id = Path(destination_folder).name
        self.environment.gpus = gpus

        configs = TrainerConfigs(
            self.data,
            self.augmentation,
            self.model,
            self.training,
            self.logging,
            self.environment,
        )

        logging_dir = train_with_yaml(
            gpus=gpus,
            data=configs.data,
            augmentation=configs.augmentation,
            model=configs.model,
            training=configs.training,
            logging=configs.logging,
            environment=configs.environment,
        )
        training_summary_path = logging_dir / "training_summary.json"
        training_summary = FileHandler.load_json(file_path=training_summary_path)
        is_success = training_summary["success"]
        status = Status.COMPLETED if is_success else Status.STOPPED

        FileHandler.remove_folder(configs.temp_folder)
        logger.info(f"Removed {configs.temp_folder} folder.")

        destination_folder = Path(self.logging.output_dir) / self.logging.project_id
        FileHandler.move_and_cleanup_folders(source_folder=logging_dir, destination_folder=destination_folder)
        logger.info(f"Files in {logging_dir} were moved to {destination_folder}.")

        best_fx_paths = list(Path(destination_folder).glob("*best_fx.pt"))
        best_onnx_paths = list(Path(destination_folder).glob("*best.onnx"))
        hparams_path = destination_folder / "hparams.yaml"

        available_options = self._get_available_options()

        if best_fx_paths:
            metadata.update_best_fx_model_path(best_fx_model_path=best_fx_paths[0].as_posix())
        if best_onnx_paths:
            metadata.update_best_onnx_model_path(best_onnx_model_path=best_onnx_paths[0].as_posix())
        metadata.update_training_result(training_summary=training_summary)
        metadata.update_hparams(hparams=hparams_path.as_posix())
        metadata.update_status(status=status)
        metadata.update_available_options(available_options)

        MetadataHandler.save_json(data=metadata.asdict(), folder_path=destination_folder)

        return metadata
