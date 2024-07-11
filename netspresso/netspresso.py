import subprocess
from typing import Optional, Union

from loguru import logger

from netspresso.benchmarker import BenchmarkerV2
from netspresso.clients.auth import TokenHandler, auth_client
from netspresso.clients.auth.response_body import UserResponse
from netspresso.clients.tao import TAOTokenHandler
from netspresso.compressor import CompressorV2
from netspresso.converter import ConverterV2
from netspresso.enums import Task
from netspresso.qai_hub.benchmarker import QAIHubBenchmarker
from netspresso.qai_hub.converter import QAIHubConverter
from netspresso.tao import TAOTrainer
from netspresso.trainer import Trainer


class NetsPresso:
    def __init__(self, email: str, password: str, verify_ssl: bool = True) -> None:
        """Initialize NetsPresso instance and perform user authentication.

        Args:
            email (str): User's email for authentication.
            password (str): User's password for authentication.
            verify_ssl (bool): Flag to indicate whether SSL certificates should be verified. Defaults to True.
        """
        self.token_handler = TokenHandler(
            email=email, password=password, verify_ssl=verify_ssl
        )
        self.user_info = self.get_user()

    def get_user(self) -> UserResponse:
        """Get user information using the access token.

        Returns:
            UserInfo: User information.
        """
        user_info = auth_client.get_user_info(
            self.token_handler.tokens.access_token, self.token_handler.verify_ssl
        )
        return user_info

    def trainer(
        self, task: Optional[Union[str, Task]] = None, yaml_path: Optional[str] = None
    ) -> Trainer:
        """Initialize and return a Trainer instance.

        Args:
            task (Union[str, Task], optional): Type of task (classification, detection, segmentation).
            yaml_path (str, optional): Path to the YAML configuration file.

        Returns:
            Trainer: Initialized Trainer instance.
        """
        return Trainer(token_handler=self.token_handler, task=task, yaml_path=yaml_path)

    def compressor_v2(self) -> CompressorV2:
        """Initialize and return a Compressor instance.

        Returns:
            Compressor: Initialized Compressor instance.
        """
        return CompressorV2(token_handler=self.token_handler)

    def converter_v2(self) -> ConverterV2:
        """Initialize and return a Converter instance.

        Returns:
            Converter: Initialized Converter instance.
        """
        return ConverterV2(token_handler=self.token_handler, user_info=self.user_info)

    def benchmarker_v2(self) -> BenchmarkerV2:
        """Initialize and return a Benchmarker instance.

        Returns:
            Benchmarker: Initialized Benchmarker instance.
        """
        return BenchmarkerV2(token_handler=self.token_handler, user_info=self.user_info)


class TAO:
    def __init__(self, ngc_api_key: str) -> None:
        """Initialize TAO instance and perform user authentication.

        Args:
            ngc_api_key (str): API key for TAO authentication.
        """
        self.ngc_api_key = ngc_api_key
        self.token_handler = TAOTokenHandler(ngc_api_key=ngc_api_key)

    def trainer(self) -> TAOTrainer:
        """Initialize and return a Trainer instance.

        Returns:
            TAO: Initialized Trainer instance.
        """
        return TAOTrainer(token_handler=self.token_handler)


class QAIHub:
    def __init__(self, api_token: str) -> None:
        # Define the command and arguments
        command = 'qai-hub'
        args = ['configure', '--api_token', f'{api_token}']

        # Execute the command
        result = subprocess.run([command] + args, capture_output=True, text=True)
        logger.info(result)

    def converter(self) -> QAIHubConverter:
        """Initialize and return a Converter instance.

        Returns:
            QAIHubConverter: Initialized Converter instance.
        """
        return QAIHubConverter()

    def benchmarker(self) -> QAIHubBenchmarker:
        """Initialize and return a Benchmarker instance.

        Returns:
            QAIHubBenchmarker: Initialized Benchmarker instance.
        """
        return QAIHubBenchmarker()
