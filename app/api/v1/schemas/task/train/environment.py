from pydantic import BaseModel, ConfigDict, Field


class EnvironmentCreate(BaseModel):
    seed: int = Field(default=1, description="Random seed to use")
    num_workers: int = Field(default=4, description="Number of workers to use")
    gpus: str = Field(default="0", description="GPUs to use")


class EnvironmentPayload(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    seed: int
    num_workers: int
    gpus: str
