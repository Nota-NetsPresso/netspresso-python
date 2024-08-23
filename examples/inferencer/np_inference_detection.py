from glob import glob
from pathlib import Path

from netspresso import NetsPresso
from netspresso.enums import Task
from netspresso.trainer.augmentations import Resize, Pad
from netspresso.trainer.optimizers import AdamW
from netspresso.trainer.schedulers import CosineAnnealingWarmRestartsWithCustomWarmUp


EMAIL = "YOUR_EMAIL"
PASSWORD = "YOUR_PASSWORD"

netspresso = NetsPresso(email=EMAIL, password=PASSWORD)

IMG_SIZE=640

# 1. Declare trainer
trainer = netspresso.trainer(task=Task.OBJECT_DETECTION)
trainer.set_dataset(dataset_root_path="YOUR_DATASET_PATH")
trainer.set_model_config(model_name="YOLOX-S", img_size=IMG_SIZE)
trainer.set_augmentation_config(
    train_transforms=[Resize(), Pad()],
    inference_transforms=[Resize(), Pad()],
)
trainer.set_training_config(
    epochs=30,
    batch_size=16,
    optimizer=AdamW(lr=6e-3),
    scheduler=CosineAnnealingWarmRestartsWithCustomWarmUp(warmup_epochs=10),
)
project_name = "project_sample"
trained_result = trainer.train(gpus="0, 1", project_name=project_name)

inferencer = netspresso.np_inferencer(config_path=trained_result.runtime)

img_path = "IMAGE_PATH"
save_path = f"./inference_results/{Path(img_path).name}"
outputs = inferencer.inference(trained_result.best_onnx_model_path, img_path, save_path)
