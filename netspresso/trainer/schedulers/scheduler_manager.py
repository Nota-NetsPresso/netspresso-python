from netspresso.enums.train import Scheduler
from netspresso.trainer.schedulers.schedulers import (
    CosineAnnealingLRWithCustomWarmUp,
    CosineAnnealingWarmRestartsWithCustomWarmUp,
    PolynomialLRWithWarmUp,
    StepLR,
)


class SchedulerManager:
    @staticmethod
    def get_scheduler(name: str):
        scheduler_map = {
            Scheduler.STEPLR: StepLR,
            Scheduler.POLYNOMIAL_LR_WITH_WARM_UP: PolynomialLRWithWarmUp,
            Scheduler.COSINE_ANNEALING_LR_WITH_CUSTOM_WARM_UP: CosineAnnealingLRWithCustomWarmUp,
            Scheduler.COSINE_ANNEALING_WARM_RESTARTS_WITH_CUSTOM_WARM_UP: CosineAnnealingWarmRestartsWithCustomWarmUp,
        }

        scheduler_class = scheduler_map.get(name.lower())
        if not scheduler_class:
            raise ValueError(f"Scheduler '{name}' not found.")

        return scheduler_class()
