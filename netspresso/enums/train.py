from enum import Enum


class Optimizer(str, Enum):
    ADADELTA = "Adadelta"
    ADAGRAD = "Adagrad"
    ADAM = "Adam"
    ADAMAX = "Adamax"
    ADAMW = "AdamW"
    RMSPROP = "RMSprop"
    SGD = "SGD"

    @classmethod
    def to_display_name(cls, name: str) -> str:
        name_map = {
            "adadelta": cls.ADADELTA,
            "adagrad": cls.ADAGRAD,
            "adam": cls.ADAM,
            "adamax": cls.ADAMAX,
            "adamw": cls.ADAMW,
            "rmsprop": cls.RMSPROP,
            "sgd": cls.SGD,
        }
        return name_map[name.lower()].value


class Scheduler(str, Enum):
    STEPLR = "StepLR"
    POLYNOMIAL_LR_WITH_WARM_UP = "PolynomialLRWithWarmUp"
    COSINE_ANNEALING_LR_WITH_CUSTOM_WARM_UP = "CosineAnnealingLRWithCustomWarmUp"
    COSINE_ANNEALING_WARM_RESTARTS_WITH_CUSTOM_WARM_UP = "CosineAnnealingWarmRestartsWithCustomWarmUp"

    @classmethod
    def to_display_name(cls, name: str) -> str:
        name_map = {
            "step": cls.STEPLR,
            "poly": cls.POLYNOMIAL_LR_WITH_WARM_UP,
            "cosine_no_sgdr": cls.COSINE_ANNEALING_LR_WITH_CUSTOM_WARM_UP,
            "cosine": cls.COSINE_ANNEALING_WARM_RESTARTS_WITH_CUSTOM_WARM_UP,
        }
        return name_map[name.lower()].value
