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
    "ResNet": ["resnet18", "resnet34", "resnet50"],
    "MobileNet": ["mobilenet_v3_small", "mobilenet_v3_large"],
    "MobileViT": ["mobilevit_s"],
    "EfficientFormer": ["efficientformer_l1"],
    "ViT": ["vit_tiny"],
    "MixNet": ["mixnet_s", "mixnet_m", "mixnet_l"],
    "YOLOX": ["yolox_s", "yolox_m", "yolox_l", "yolox_x"],
    "YOLO": ["yolo_fastest"],
    "SegFormer": ["segformer_b0"],
    "PIDNet": ["pidnet_s"],
}

MODEL_NAME_DISPLAY_MAP = {
    "efficientformer_l1": "EfficientFormer-L1",
    "mobilenet_v3_small": "MobileNetV3-S",
    "mobilenet_v3_large": "MobileNetV3-L",
    "mobilevit_s": "MobileViT-S",
    "resnet18": "ResNet18",
    "resnet34": "ResNet34",
    "resnet50": "ResNet50",
    "vit_tiny": "ViT-T",
    "mixnet_s": "MixNet-S",
    "mixnet_m": "MixNet-M",
    "mixnet_l": "MixNet-L",
    "yolox_s": "YOLOX-S",
    "yolox_m": "YOLOX-M",
    "yolox_l": "YOLOX-L",
    "yolox_x": "YOLOX-X",
    "yolo_fastest": "YOLO-Fastest",
    "segformer_b0": "SegFormer-B0",
    "pidnet_s": "PIDNet-S",
}

TASK_NAME_DISPLAY_MAP = {
    "classification": "Classification",
    "detection": "Object Detection",
    "segmentation": "Semantic Segmentation",
}

FRAMEWORK_NAME_DISPLAY_MAP = {
    "pytorch": "PyTorch",
}

CLASSIFICATION_MODELS = {
    "efficientformer_l1": ClassificationEfficientFormerModelConfig,
    "mobilenet_v3_small": ClassificationMobileNetV3SmallModelConfig,
    "mobilenet_v3_large": ClassificationMobileNetV3LargeModelConfig,
    "mobilevit_s": ClassificationMobileViTModelConfig,
    "resnet18": ClassificationResNet18ModelConfig,
    "resnet34": ClassificationResNet34ModelConfig,
    "resnet50": ClassificationResNet50ModelConfig,
    "vit_tiny": ClassificationViTTinyModelConfig,
    "mixnet_s": ClassificationMixNetSmallModelConfig,
    "mixnet_m": ClassificationMixNetMediumModelConfig,
    "mixnet_l": ClassificationMixNetLargeModelConfig,
}

DETECTION_MODELS = {
    "yolox_s": DetectionYoloXSModelConfig,
    "yolox_m": DetectionYoloXMModelConfig,
    "yolox_l": DetectionYoloXLModelConfig,
    "yolox_x": DetectionYoloXXModelConfig,
    "yolo_fastest": DetectionYoloFastestModelConfig,
}

SEGMENTATION_MODELS = {
    "efficientformer_l1": SegmentationEfficientFormerModelConfig,
    "mobilenet_v3_small": SegmentationMobileNetV3SmallModelConfig,
    "resnet50": SegmentationResNet50ModelConfig,
    "segformer_b0": SegmentationSegFormerB0ModelConfig,
    "mixnet_s": SegmentationMixNetSmallModelConfig,
    "mixnet_m": SegmentationMixNetMediumModelConfig,
    "mixnet_l": SegmentationMixNetLargeModelConfig,
    "pidnet_s": PIDNetModelConfig,
}

POSEESTIMATION_MODELS = {
    "mobilenet_v3_small": PoseEstimationMobileNetV3SmallModelConfig,
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

def get_model_display_name(model_name: str) -> str:
    """Get the display name for a given model name.

    Args:
        model_name (str): Name of the model

    Returns:
        str: Display name of the model. Returns None if model is not found.
    """
    return MODEL_NAME_DISPLAY_MAP.get(model_name, None)
