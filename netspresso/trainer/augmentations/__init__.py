from netspresso.trainer.augmentations.augmentation import (
    AugmentationConfig,
    ClassificationAugmentationConfig,
    ColorJitter,
    DetectionAugmentationConfig,
    Inference,
    Pad,
    RandomCrop,
    RandomCutmix,
    RandomHorizontalFlip,
    RandomMixup,
    RandomResizedCrop,
    RandomVerticalFlip,
    Resize,
    SegmentationAugmentationConfig,
    Train,
    Transform,
    TrivialAugmentWide,
)

AUGMENTATION_CONFIG_TYPE = {
    "classification": ClassificationAugmentationConfig,
    "detection": DetectionAugmentationConfig,
    "segmentation": SegmentationAugmentationConfig,
}


__all__ = [
    "ColorJitter",
    "Pad",
    "RandomCrop",
    "RandomResizedCrop",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "Resize",
    "TrivialAugmentWide",
    "RandomMixup",
    "RandomCutmix",
    "Inference",
    "Train",
    "Transform",
    "AugmentationConfig",
    "AUGMENTATION_CONFIG_TYPE",
]
