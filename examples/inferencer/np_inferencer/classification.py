from pathlib import Path
from glob import glob

from netspresso import NetsPresso
from netspresso.enums import Task
from netspresso.trainer.augmentations import Resize, Pad
from netspresso.trainer.optimizers import AdamW
from netspresso.trainer.schedulers import CosineAnnealingWarmRestartsWithCustomWarmUp


EMAIL = "YOUR_EMAIL"
PASSWORD = "YOUR_PASSWORD"

netspresso = NetsPresso(email=EMAIL, password=PASSWORD)
trainer = netspresso.trainer(task=Task.IMAGE_CLASSIFICATION)
trainer.set_dataset(dataset_root_path="/root/datasets/cifar100")
trainer.set_model_config(model_name="MobileNetV3-S", img_size=128)
trainer.set_augmentation_config(
    train_transforms=[Resize(), Pad()],
    inference_transforms=[Resize(), Pad()],
)
trainer.set_training_config(
    epochs=50,
    batch_size=512,
    optimizer=AdamW(lr=6e-3),
    scheduler=CosineAnnealingWarmRestartsWithCustomWarmUp(warmup_epochs=10),
)
trained_result = trainer.train(gpus="0, 1", project_name="classification")

# 1. Declare inferencer
config_path = trained_result.runtime
input_model_path = trained_result.best_onnx_model_path
inferencer = netspresso.np_inferencer(config_path=config_path, input_model_path=input_model_path)

# 2. Inference image
valid_imgs = glob("/root/datasets/cifar100/images/valid/*.png")
for valid_img in valid_imgs[:100]:
    save_path = f"{Path(input_model_path).parent}/inference_results/{Path(valid_img).name}"
    outputs = inferencer.inference(image_path=valid_img, save_path=save_path)
