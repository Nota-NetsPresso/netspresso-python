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
from netspresso.trainer.models.yolox import (
    DetectionYoloXLModelConfig,
    DetectionYoloXMModelConfig,
    DetectionYoloXSModelConfig,
    DetectionYoloXXModelConfig,
)

CLASSIFICATION_MODELS = {
    "EfficientFormer": ClassificationEfficientFormerModelConfig,
    "MobileNetV3_Small": ClassificationMobileNetV3SmallModelConfig,
    "MobileNetV3-S": ClassificationMobileNetV3SmallModelConfig,
    "MobileNetV3_Large": ClassificationMobileNetV3LargeModelConfig,
    "MobileNetV3-L": ClassificationMobileNetV3LargeModelConfig,
    "MobileViT": ClassificationMobileViTModelConfig,
    "MobileViT-S": ClassificationMobileViTModelConfig,
    "ResNet18": ClassificationResNet18ModelConfig,
    "ResNet34": ClassificationResNet34ModelConfig,
    "ResNet50": ClassificationResNet50ModelConfig,
    "ViT-Tiny": ClassificationViTTinyModelConfig,
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
}

SEGMENTATION_MODELS = {
    "EfficientFormer": SegmentationEfficientFormerModelConfig,
    "MobileNetV3_Small": SegmentationMobileNetV3SmallModelConfig,
    "MobileNetV3-S": SegmentationMobileNetV3SmallModelConfig,
    "ResNet50": SegmentationResNet50ModelConfig,
    "SegFormer-B0": SegmentationSegFormerB0ModelConfig,
    "MixNet-S": SegmentationMixNetSmallModelConfig,
    "MixNet-M": SegmentationMixNetMediumModelConfig,
    "MixNet-L": SegmentationMixNetLargeModelConfig,
    "PIDNet": PIDNetModelConfig,
}

POSEESTIMATION_MODELS = {
    "MobileNetV3_Small": PoseEstimationMobileNetV3SmallModelConfig,
    "MobileNetV3-S": PoseEstimationMobileNetV3SmallModelConfig,
}


__all__ = [
    "CLASSIFICATION_MODELS",
    "DETECTION_MODELS",
    "SEGMENTATION_MODELS",
    "CheckpointConfig",
    "ModelConfig",
]
