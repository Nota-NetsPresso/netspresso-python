from typing import Any, Dict, List

from sqlalchemy.orm import Session

from app.api.v1.schemas.model import ModelPayload
from app.api.v1.schemas.task.train.train_task import TrainingCreate, TrainTaskSchema
from app.services.user import user_service
from netspresso.trainer.augmentations.augmentation import Normalize, Resize, ToTensor
from netspresso.trainer.models import get_all_available_models
from netspresso.trainer.optimizers.optimizer_manager import OptimizerManager
from netspresso.trainer.optimizers.optimizers import get_supported_optimizers
from netspresso.trainer.schedulers.scheduler_manager import SchedulerManager
from netspresso.trainer.schedulers.schedulers import get_supported_schedulers
from netspresso.utils.db.repositories.task import train_task_repository


class TaskService:
    def get_supported_models(self) -> Dict[str, List[str]]:
        return get_all_available_models()

    def get_supported_optimizers(self) -> List[Dict[str, Any]]:
        return get_supported_optimizers()

    def get_supported_schedulers(self) -> List[Dict[str, Any]]:
        return get_supported_schedulers()

    def create_train_task(self, db: Session, training_in: TrainingCreate, api_key: str) -> ModelPayload:
        netspresso = user_service.build_netspresso_with_api_key(db=db, api_key=api_key)

        trainer = netspresso.trainer(task=training_in.task)
        trainer.set_dataset_config(
            name="test",
            root_path=training_in.dataset.root_path,
            train_image="train/images",
            train_label="train/labels",
            valid_image="valid/images",
            valid_label="valid/labels",
            id_mapping=training_in.dataset.id_mapping,
        )
        img_size = training_in.input_shapes[0].dimension[0]
        trainer.set_model_config(model_name=training_in.pretrained_model_name, img_size=img_size)
        trainer.set_augmentation_config(
            train_transforms=[Resize(), ToTensor(), Normalize()],
            inference_transforms=[Resize(), ToTensor(), Normalize()],
        )
        optimizer = OptimizerManager.get_optimizer(
            name=training_in.hyperparameter.optimizer.name,
            lr=training_in.hyperparameter.learning_rate,
        )
        scheduler = SchedulerManager.get_scheduler(name=training_in.hyperparameter.scheduler.name)

        trainer.set_training_config(
            epochs=training_in.hyperparameter.epochs,
            batch_size=training_in.hyperparameter.batch_size,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        trained_model = trainer.train(
            gpus=training_in.environment.gpus,
            model_name=training_in.name,
            project_id=training_in.project_id,
        )

        return trained_model

    def get_task(self, db: Session, task_id: str, api_key: str) -> TrainTaskSchema:
        netspresso = user_service.build_netspresso_with_api_key(db=db, api_key=api_key)

        train_task = train_task_repository.get_by_task_id(db=db, task_id=task_id, user_id=netspresso.user_info.user_id)
        train_task = TrainTaskSchema.model_validate(train_task)

        return train_task


task_service = TaskService()
