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
trainer = netspresso.trainer(task=Task.OBJECT_DETECTION)
trainer.set_dataset(dataset_root_path="YOUR_DATASET_PATH")
trainer.set_model_config(model_name="YOLOX-S", img_size=640)
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
trained_result = trainer.train(gpus="0, 1", project_name="detection")

# 1. Declare inferencer
config_path = trained_result.runtime
inferencer = netspresso.np_inferencer(config_path=config_path)

# 2. Inference image
valid_imgs = glob("/root/datasets/traffic-sign/images/valid/*.jpg")
for valid_img in valid_imgs:
    input_model_path = trained_result.best_onnx_model_path
    save_path = f"{Path(input_model_path).parent}/inference_results/{Path(valid_img).name}"
    outputs = inferencer.inference(input_model_path, valid_img, save_path)
