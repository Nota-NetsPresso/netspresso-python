import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    API_PREFIX: str = "/api/v1"
    SERVER_ADDRESS: str = os.environ.get("SERVER_ADDRESS", "0.0.0.0")
    SERVER_PORT: int = int(os.environ.get("SERVER_PORT", 80))
    SERVER_WORKERS: int = int(os.environ.get("SERVER_WORKERS", 1))
    SERVER_URL: str = f"{SERVER_ADDRESS}:{SERVER_PORT}{API_PREFIX}"


settings = Settings()
