from typing import List

from pydantic import BaseModel, ConfigDict


class PerformancePayload(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    train_losses: dict
    valid_losses: dict
    train_metrics: dict
    valid_metrics: dict
    metrics_list: List[str]
    primary_metric: str
    flops: int
    params: int
    total_train_time: float
    best_epoch: int
    last_epoch: int
    total_epoch: int
    status: str
