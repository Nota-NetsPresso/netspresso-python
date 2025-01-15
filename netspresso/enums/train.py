from enum import Enum


class Optimizer(str, Enum):
    ADADELTA = "adadelta"
    ADAGRAD = "adagrad"
    ADAM = "adam"
    ADAMAX = "adamax"
    ADAMW = "adamw"
    RMSPROP = "rmsprop"
    SGD = "sgd"

    @classmethod
    def to_display_name(cls, name: str) -> str:
        name_map = {
            "adadelta": "Adadelta",
            "adagrad": "Adagrad",
            "adam": "Adam",
            "adamax": "Adamax",
            "adamw": "AdamW",
            "rmsprop": "RMSprop",
            "sgd": "SGD",
        }
        return name_map[name.lower()]


class Scheduler(str, Enum):
    STEPLR = "step"
    POLYNOMIAL_LR_WITH_WARM_UP = "poly"
    COSINE_ANNEALING_LR_WITH_CUSTOM_WARM_UP = "cosine_no_sgdr"
    COSINE_ANNEALING_WARM_RESTARTS_WITH_CUSTOM_WARM_UP = "cosine"

    @classmethod
    def to_display_name(cls, name: str) -> str:
        name_map = {
            "step": "StepLR",
            "poly": "PolynomialLRWithWarmUp",
            "cosine_no_sgdr": "CosineAnnealingLRWithCustomWarmUp",
            "cosine": "CosineAnnealingWarmRestartsWithCustomWarmUp",
        }
        return name_map[name.lower()]


class StorageLocation(str, Enum):
    LOCAL = "local"
    STORAGE = "storage"


class AugmentationType(str, Enum):
    train = "train"
    inference = "inference"
