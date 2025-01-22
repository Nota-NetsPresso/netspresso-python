from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class DatasetCreate(BaseModel):
    train_path: Optional[str] = None
    valid_path: Optional[str] = None
    test_path: Optional[str] = None


class DatasetPayload(BaseModel):
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
