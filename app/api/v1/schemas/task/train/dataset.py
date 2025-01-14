from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DatasetCreate(BaseModel):
    root_path: Optional[str] = None
    train_path: Optional[str] = None
    valid_path: Optional[str] = None
    test_path: Optional[str] = None
    storage_location: str
    train_valid_split_ratio: float = Field(default=0.0)
    id_mapping: Optional[List] = []

    @model_validator(mode="after")
    def check_paths(cls, values):
        root_path = values.get('root_path')
        train_path = values.get('train_path')
        id_mapping = values.get('id_mapping')

        if not root_path:
            if not train_path:
                raise ValueError('train_path must be provided if root_path is not set.')
            if not id_mapping:
                raise ValueError('id_mapping must be provided if root_path is not set.')
        return values


class DatasetSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    format: str
    root_path: str
    train_path: str
    valid_path: Optional[str]
    test_path: Optional[str]
    storage_location: str
    train_valid_split_ratio: float
    id_mapping: Optional[List] = []
    palette: Optional[dict] = {}
