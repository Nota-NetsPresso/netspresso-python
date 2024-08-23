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
trainer = netspresso.trainer(task=Task.SEMANTIC_SEGMENTATION)
trainer.set_dataset(dataset_root_path="/root/datasets/voc2012_seg")
trainer.set_model_config(model_name="MobileNetV3-S", img_size=224)
trainer.set_augmentation_config(
    train_transforms=[Resize(), Pad(fill=114)],
    inference_transforms=[Resize(), Pad(fill=114)],
)
trainer.set_training_config(
    epochs=100,
    batch_size=64,
    optimizer=AdamW(lr=6e-2),
    scheduler=CosineAnnealingWarmRestartsWithCustomWarmUp(warmup_epochs=10),
)
trained_result = trainer.train(gpus="0, 1", project_name="segmentation")

# 1. Declare inferencer
config_path = trained_result.runtime
inferencer = netspresso.np_inferencer(config_path=config_path)

# 2. Inference image
valid_imgs = glob("/root/datasets/voc2012_seg/images/valid/*.jpg")
for valid_img in valid_imgs[:100]:
    input_model_path = trained_result.best_onnx_model_path
    save_path = Path(input_model_path).parent / "inference_results" / Path(valid_img).name
    outputs = inferencer.inference(input_model_path, valid_img, save_path)
