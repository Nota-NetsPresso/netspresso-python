from enum import Enum


class SubFolder(str, Enum):
    TRAINED_MODELS = "Trained models"
    COMPRESSED_MODELS = "Compressed models"
    PRETRAINED_MODELS = "Pretrained models"
