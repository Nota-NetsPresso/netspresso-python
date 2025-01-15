from netspresso.enums.train import Optimizer
from netspresso.trainer.optimizers.optimizers import (
    SGD,
    Adadelta,
    Adagrad,
    Adam,
    Adamax,
    AdamW,
    RMSprop,
)


class OptimizerManager:
    @staticmethod
    def get_optimizer(optimizer_name: str):
        optimizer_map = {
            Optimizer.ADADELTA: Adadelta,
            Optimizer.ADAGRAD: Adagrad,
            Optimizer.ADAM: Adam,
            Optimizer.ADAMAX: Adamax,
            Optimizer.ADAMW: AdamW,
            Optimizer.RMSPROP: RMSprop,
            Optimizer.SGD: SGD,
        }

        optimizer_class = optimizer_map.get(optimizer_name.lower())
        if not optimizer_class:
            raise ValueError(f"Optimizer '{optimizer_name}' not found.")

        return optimizer_class()
