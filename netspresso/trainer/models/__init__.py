from typing import Dict, List

from netspresso.trainer.models.base import CheckpointConfig, ModelConfig
from netspresso.trainer.models.efficientformer import (
    ClassificationEfficientFormerModelConfig,
    DetectionEfficientFormerModelConfig,
    SegmentationEfficientFormerModelConfig,
)
from netspresso.trainer.models.mixnet import (
    ClassificationMixNetLargeModelConfig,
    ClassificationMixNetMediumModelConfig,
    ClassificationMixNetSmallModelConfig,
    DetectionMixNetLargeModelConfig,
    DetectionMixNetMediumModelConfig,
    DetectionMixNetSmallModelConfig,
    SegmentationMixNetLargeModelConfig,
    SegmentationMixNetMediumModelConfig,
    SegmentationMixNetSmallModelConfig,
)
from netspresso.trainer.models.mobilenetv3 import (
    ClassificationMobileNetV3LargeModelConfig,
    ClassificationMobileNetV3SmallModelConfig,
    DetectionMobileNetV3SmallModelConfig,
    SegmentationMobileNetV3SmallModelConfig,
)
from netspresso.trainer.models.mobilevit import ClassificationMobileViTModelConfig
from netspresso.trainer.models.pidnet import PIDNetModelConfig
from netspresso.trainer.models.resnet import (
    ClassificationResNet18ModelConfig,
    ClassificationResNet34ModelConfig,
    ClassificationResNet50ModelConfig,
    DetectionResNet50ModelConfig,
    SegmentationResNet50ModelConfig,
)
from netspresso.trainer.models.rtmpose import PoseEstimationMobileNetV3SmallModelConfig
from netspresso.trainer.models.segformer import SegmentationSegFormerB0ModelConfig
from netspresso.trainer.models.vit import ClassificationViTTinyModelConfig
from netspresso.trainer.models.yolo import DetectionYoloFastestModelConfig
from netspresso.trainer.models.yolox import (
    DetectionYoloXLModelConfig,
    DetectionYoloXMModelConfig,
    DetectionYoloXSModelConfig,
    DetectionYoloXXModelConfig,
)

MODEL_GROUPS = {
    "ResNet": ["ResNet18", "ResNet34", "ResNet50"],
    "MobileNet": ["MobileNetV3-S", "MobileNetV3-L"],
    "MobileViT": ["MobileViT-S"],
    "EfficientFormer": ["EfficientFormer-L1"],
    "ViT": ["ViT-T"],
    "MixNet": ["MixNet-S", "MixNet-M", "MixNet-L"],
    "YOLOX": ["YOLOX-S", "YOLOX-M", "YOLOX-L", "YOLOX-X"],
    "SegFormer": ["SegFormer-B0"],
    "PIDNet": ["PIDNet-S"],
}

MODEL_NAME_DISPLAY_MAP = {
    "EfficientFormer-L1": "efficientformer_l1",
    "MobileNetV3-S": "mobilenet_v3_small",
    "MobileNetV3-L": "mobilenet_v3_large",
    "MobileViT-S": "mobilevit_s",
    "ResNet18": "resnet18",
    "ResNet34": "resnet34",
    "ResNet50": "resnet50",
    "ViT-T": "vit_tiny",
    "MixNet-S": "mixnet_s",
    "MixNet-M": "mixnet_m",
    "MixNet-L": "mixnet_l",
    "YOLOX-S": "yolox_s",
    "YOLOX-M": "yolox_m",
    "YOLOX-L": "yolox_l",
    "YOLOX-X": "yolox_x",
    "SegFormer-B0": "segformer_b0",
    "PIDNet-S": "pidnet_s",
}

CLASSIFICATION_MODELS = {
    "EfficientFormer-L1": ClassificationEfficientFormerModelConfig,
    "MobileNetV3-S": ClassificationMobileNetV3SmallModelConfig,
    "MobileNetV3-L": ClassificationMobileNetV3LargeModelConfig,
    "MobileViT-S": ClassificationMobileViTModelConfig,
    "ResNet18": ClassificationResNet18ModelConfig,
    "ResNet34": ClassificationResNet34ModelConfig,
    "ResNet50": ClassificationResNet50ModelConfig,
    "ViT-T": ClassificationViTTinyModelConfig,
    "MixNet-S": ClassificationMixNetSmallModelConfig,
    "MixNet-M": ClassificationMixNetMediumModelConfig,
    "MixNet-L": ClassificationMixNetLargeModelConfig,
}

DETECTION_MODELS = {
    "YOLOX-S": DetectionYoloXSModelConfig,
    "YOLOX-M": DetectionYoloXMModelConfig,
    "YOLOX-L": DetectionYoloXLModelConfig,
    "YOLOX-X": DetectionYoloXXModelConfig,
    # "YOLO-Fastest": DetectionYoloFastestModelConfig,
}

SEGMENTATION_MODELS = {
    "EfficientFormer-L1": SegmentationEfficientFormerModelConfig,
    "MobileNetV3-S": SegmentationMobileNetV3SmallModelConfig,
    "ResNet50": SegmentationResNet50ModelConfig,
    "SegFormer-B0": SegmentationSegFormerB0ModelConfig,
    "MixNet-S": SegmentationMixNetSmallModelConfig,
    "MixNet-M": SegmentationMixNetMediumModelConfig,
    "MixNet-L": SegmentationMixNetLargeModelConfig,
    "PIDNet-S": PIDNetModelConfig,
}

POSEESTIMATION_MODELS = {
    "MobileNetV3-S": PoseEstimationMobileNetV3SmallModelConfig,
}

# NOT_SUPPORTED_PRETRAINED_MODELS = ["YOLO-Fastest"]


__all__ = [
    "CLASSIFICATION_MODELS",
    "DETECTION_MODELS",
    "SEGMENTATION_MODELS",
    "POSEESTIMATION_MODELS",
    "MODEL_NAME_DISPLAY_MAP",
    "CheckpointConfig",
    "ModelConfig",
]


def get_all_available_models() -> Dict[str, List[str]]:
    """Get all available models for each task, excluding deprecated names.

    Returns:
        Dict[str, List[str]]: A dictionary mapping each task to its available models.
    """
    all_models = {
        "classification": list(CLASSIFICATION_MODELS),
        "detection": list(DETECTION_MODELS),
        "segmentation": list(SEGMENTATION_MODELS),
    }
    return all_models

def get_model_group(model_name: str) -> str:
    """Get the group name for a given model name.

    Args:
        model_name (str): Name of the model

    Returns:
        str: Group name of the model. Returns None if model is not found in any group.
    """
    for group_name, models in MODEL_GROUPS.items():
        if model_name in models:
            return group_name
    return None
