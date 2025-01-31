from pydantic import BaseModel, ConfigDict


class AugmentationPayload(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    parameters: dict
    phase: str
    hyperparameter_id: int
