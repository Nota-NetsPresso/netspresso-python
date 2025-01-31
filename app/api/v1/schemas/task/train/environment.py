from pydantic import BaseModel, ConfigDict, Field


class EnvironmentCreate(BaseModel):
    gpus: str = Field(default="0", description="GPUs to use")


class EnvironmentPayload(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    gpus: str
