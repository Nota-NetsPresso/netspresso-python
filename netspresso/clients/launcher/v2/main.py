from netspresso.clients.config import Config, Module
from netspresso.clients.launcher.v2.benchmarker import Benchmarker
from netspresso.clients.launcher.v2.converter import Converter


class LauncherAPIClient:
    def __init__(self):
        self.config = Config(Module.LAUNCHER)
        self.host = self.config.HOST
        self.port = self.config.PORT
        self.prefix = self.config.URI_PREFIX
        self.url = f"{self.host}:{self.port}{self.prefix}"
        self.converter = self._create_convert_api()
        self.benchmarker = self._create_benchmark_api()

    def _create_convert_api(self):
        return Converter(self.url)

    def _create_benchmark_api(self):
        return Benchmarker(self.url)

    def is_cloud(self) -> bool:
        return self.config.is_cloud()


launcher_client_v2 = LauncherAPIClient()
