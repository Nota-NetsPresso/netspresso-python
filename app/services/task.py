from typing import Any, Dict, List

from sqlalchemy.orm import Session

from app.api.v1.schemas.model import ModelPayload
from app.api.v1.schemas.task.train.hyperparameter import TrainerModel
from app.api.v1.schemas.task.train.train_task import TrainingCreate, TrainingPayload
from app.services.user import user_service
from netspresso.trainer.augmentations.augmentation import Normalize, Resize, ToTensor
from netspresso.trainer.models import MODEL_NAME_DISPLAY_MAP, get_all_available_models, get_model_group
from netspresso.trainer.optimizers.optimizer_manager import OptimizerManager
from netspresso.trainer.optimizers.optimizers import get_supported_optimizers
from netspresso.trainer.schedulers.scheduler_manager import SchedulerManager
from netspresso.trainer.schedulers.schedulers import get_supported_schedulers
from netspresso.utils.db.repositories.task import train_task_repository


class TaskService:
    def get_supported_models(self) -> Dict[str, List[str]]:
        available_models = get_all_available_models()

        for task, models in available_models.items():
            available_models[task] = [
                TrainerModel(
                    name=MODEL_NAME_DISPLAY_MAP[model],
                    display_name=model,
                    group_name=get_model_group(model),
                ) for model in models
            ]

        return available_models

    def get_supported_optimizers(self) -> List[Dict[str, Any]]:
        return get_supported_optimizers()

    def get_supported_schedulers(self) -> List[Dict[str, Any]]:
        return get_supported_schedulers()

    def create_training_task(self, db: Session, training_in: TrainingCreate, api_key: str) -> TrainingPayload:
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
            name=training_in.hyperparameter.optimizer,
            lr=training_in.hyperparameter.learning_rate,
        )
        scheduler = SchedulerManager.get_scheduler(name=training_in.hyperparameter.scheduler)

        trainer.set_training_config(
            epochs=training_in.hyperparameter.epochs,
            batch_size=training_in.hyperparameter.batch_size,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        training_task = trainer.train(
            gpus=training_in.environment.gpus,
            model_name=training_in.name,
            project_id=training_in.project_id,
        )

        return training_task

    def get_task(self, db: Session, task_id: str, api_key: str) -> TrainingPayload:
        netspresso = user_service.build_netspresso_with_api_key(db=db, api_key=api_key)

        train_task = train_task_repository.get_by_task_id(db=db, task_id=task_id)
        train_task.hyperparameter.optimizer = train_task.hyperparameter.optimizer.name
        train_task.hyperparameter.scheduler = train_task.hyperparameter.scheduler.name
        model_id = train_task.model.model_id
        train_task = TrainingPayload.model_validate(train_task)
        train_task.model_id = model_id

        return train_task


task_service = TaskService()
