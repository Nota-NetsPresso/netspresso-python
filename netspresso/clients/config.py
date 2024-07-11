import configparser
import os
from enum import Enum
from pathlib import Path

from loguru import logger

BASE_DIR = Path(__file__).resolve().parent
config_parser = configparser.ConfigParser()
DEPLOYMENT_MODE = os.getenv("DEPLOYMENT_MODE", "v2-prod-cloud")
logger.info(f"Read {DEPLOYMENT_MODE} config")
config_parser.read(f"{BASE_DIR}/configs/config-{DEPLOYMENT_MODE.lower()}.ini")


class EnvironmentType(str, Enum):
    V2_PROD_CLOUD = "v2-prod-cloud"
    V2_PROD_ON_PREM = "v2-prod-on-prem"
    V2_STAGING_CLOUD = "v2-staging-cloud"
    V2_STAGING_ON_PREM = "v2-staging-on-prem"
    V2_DEV_CLOUD = "v2-dev-cloud"
    V2_DEV_ON_PREM = "v2-dev-on-prem"


class Module(str, Enum):
    GENERAL = "GENERAL"
    COMPRESSOR = "COMPRESSOR"
    LAUNCHER = "LAUNCHER"
    TAO = "TAO"


class EndPointProperty(str, Enum):
    HOST = "HOST"
    PORT = "PORT"
    URI_PREFIX = "URI_PREFIX"


class Config:
    def __init__(self, module: Module = Module.GENERAL):
        self.ENVIRONMENT_TYPE = EnvironmentType(DEPLOYMENT_MODE.lower())
        self.MODULE = module
        self.HOST = config_parser[self.MODULE][EndPointProperty.HOST]
        self.PORT = int(config_parser[self.MODULE][EndPointProperty.PORT])
        self.URI_PREFIX = config_parser[self.MODULE][EndPointProperty.URI_PREFIX]

    def is_cloud(self) -> bool:
        return self.ENVIRONMENT_TYPE in [EnvironmentType.V2_PROD_CLOUD, EnvironmentType.V2_STAGING_CLOUD, EnvironmentType.V2_DEV_CLOUD]
