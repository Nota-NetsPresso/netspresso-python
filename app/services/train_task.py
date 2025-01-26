from typing import Any, Dict, List
from pathlib import Path

from sqlalchemy.orm import Session

from app.api.v1.schemas.task.train.hyperparameter import (
    TrainerModel, 
    OptimizerPayload, 
    SchedulerPayload
)
from app.api.v1.schemas.task.train.train_task import (
    TrainingCreate, 
    TrainingPayload, 
    PretrainedModelPayload, 
    TaskPayload, 
    FrameworkPayload
)
from app.services.user import user_service
from netspresso.enums.train import Optimizer, Scheduler
from netspresso.trainer.augmentations.augmentation import Normalize, Resize, ToTensor
from netspresso.trainer.models import get_all_available_models
from netspresso.enums.train import (
    MODEL_DISPLAY_MAP,
    MODEL_GROUP_MAP,
    Optimizer,
    OptimizerDisplay,
    Scheduler,
    SchedulerDisplay,
    OPTIMIZER_DISPLAY_MAP,
    SCHEDULER_DISPLAY_MAP,
    TASK_DISPLAY_MAP,
    FRAMEWORK_DISPLAY_MAP,
)
from netspresso.trainer.optimizers.optimizer_manager import OptimizerManager
from netspresso.trainer.optimizers.optimizers import get_supported_optimizers
from netspresso.trainer.schedulers.scheduler_manager import SchedulerManager
from netspresso.trainer.schedulers.schedulers import get_supported_schedulers
from netspresso.trainer.trainer import Trainer
from netspresso.utils.db.models.train import TrainTask
from netspresso.utils.db.repositories.task import train_task_repository


class TrainTaskService:
    def get_supported_models(self) -> Dict[str, List[TrainerModel]]:
        """Get all supported models grouped by task."""
        available_models = get_all_available_models()
        supported_models = {
            task: [
                TrainerModel(
                    name=model,
                    display_name=MODEL_DISPLAY_MAP.get(model),
                    group_name=MODEL_GROUP_MAP.get(model),
                ) for model in models
            ] for task, models in available_models.items()
        }

        return supported_models

    def get_supported_optimizers(self) -> List[OptimizerPayload]:
        """Get all supported optimizers."""
        supported_optimizers = get_supported_optimizers()
        optimizers = [
            OptimizerPayload(name=optimizer.get("name"))
            for optimizer in supported_optimizers
        ]
        return optimizers

    def get_supported_schedulers(self) -> List[SchedulerPayload]:
        """Get all supported schedulers."""
        supported_schedulers = get_supported_schedulers()
        schedulers = [
            SchedulerPayload(name=scheduler.get("name"))
            for scheduler in supported_schedulers
        ]
        return schedulers

    def _setup_trainer(self, trainer, training_in: TrainingCreate) -> Trainer:
        """Configure trainer with the given training parameters."""
        trainer.set_dataset_config(
            name="test",
            root_path="/root/projects/traffic-sign",
            train_image="train/images",
            train_label="train/labels",
            valid_image="valid/images",
            valid_label="valid/labels",
            id_mapping=["prohibitory", "danger", "mandatory", "other"],
        )

        img_size = training_in.input_shapes[0].dimension[0]
        trainer.set_model_config(
            model_name=training_in.pretrained_model,
            img_size=img_size
        )

        trainer.set_augmentation_config(
            train_transforms=[Resize(), ToTensor(), Normalize()],
            inference_transforms=[Resize(), ToTensor(), Normalize()],
        )

        optimizer = OptimizerManager.get_optimizer(
            name=training_in.hyperparameter.optimizer,
            lr=training_in.hyperparameter.learning_rate,
        )
        scheduler = SchedulerManager.get_scheduler(
            name=training_in.hyperparameter.scheduler
        )

        trainer.set_training_config(
            epochs=training_in.hyperparameter.epochs,
            batch_size=training_in.hyperparameter.batch_size,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        return trainer

    def _convert_to_payload_format(self, training_task: TrainTask) -> TrainingPayload:
        """Convert training task to payload format."""
        # Set task information
        training_task.task = TaskPayload(name=training_task.task)
        training_task.framework = FrameworkPayload(name=training_task.framework)
        training_task.pretrained_model = PretrainedModelPayload(name=training_task.pretrained_model)

        # Set hyperparameter information
        training_task.hyperparameter.learning_rate = training_task.hyperparameter.optimizer["lr"]
        training_task.hyperparameter.optimizer = OptimizerPayload(name=training_task.hyperparameter.optimizer["name"])
        training_task.hyperparameter.scheduler = SchedulerPayload(name=training_task.hyperparameter.scheduler["name"])

        # Set model ID
        training_task.model_id = training_task.model.model_id

        return TrainingPayload.model_validate(training_task)

    def _generate_unique_model_name(self, db: Session, project_id: str, name: str, api_key: str) -> str:
        """Generate a unique model name by adding numbering if necessary.
        
        Args:
            project_id (str): Project ID to check existing models
            base_name (str): Original model name
            
        Returns:
            str: Unique model name with numbering if needed
        """
        netspresso = user_service.build_netspresso_with_api_key(db=db, api_key=api_key)
        project = netspresso.get_project(project_id=project_id)
        models_dir = Path(project.project_abs_path) / "trained_models"
        
        if not models_dir.exists():
            return name
            
        existing_names = [d.name for d in models_dir.iterdir() if d.is_dir()]
        
        if name not in existing_names:
            return name
            
        counter = 1
        while f"{name} ({counter})" in existing_names:
            counter += 1
            
        return f"{name} ({counter})"

    def create_training_task(self, db: Session, training_in: TrainingCreate, api_key: str) -> TrainingPayload:
        """Create and execute a new training task."""
        netspresso = user_service.build_netspresso_with_api_key(db=db, api_key=api_key)
        trainer = netspresso.trainer(task=training_in.task)
        
        trainer = self._setup_trainer(trainer, training_in)
        
        unique_model_name = self._generate_unique_model_name(
            db=db,
            project_id=training_in.project_id,
            name=training_in.name,
            api_key=api_key,
        )
        
        training_task = trainer.train(
            gpus=training_in.environment.gpus,
            model_name=unique_model_name,
            project_id=training_in.project_id,
        )

        return self._convert_to_payload_format(training_task)

    def get_training_task(self, db: Session, task_id: str, api_key: str) -> TrainingPayload:
        """Get training task by task ID."""
        netspresso = user_service.build_netspresso_with_api_key(db=db, api_key=api_key)
        training_task = train_task_repository.get_by_task_id(db=db, task_id=task_id)

        return self._convert_to_payload_format(training_task)


train_task_service = TrainTaskService()