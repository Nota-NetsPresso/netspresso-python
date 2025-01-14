from pydantic import BaseModel, ConfigDict


class AugmentationSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    parameters: dict
    phase: str
    hyperparameter_id: int
