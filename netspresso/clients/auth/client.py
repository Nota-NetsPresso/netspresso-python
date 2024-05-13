from datetime import datetime

import jwt
import pytz
from loguru import logger

from netspresso.clients.auth.response_body import TokenResponse, UserResponse
from netspresso.clients.auth.v1.client import AuthClientV1
from netspresso.clients.auth.v2.client import AuthClientV2
from netspresso.clients.config import Config, Module


class AuthClient:
    def __init__(self, config: Config = Config(Module.GENERAL)):
        """Initialize the UserSession.

        Args:
            email (str): The email address for a user account.
            password (str): The password for a user account.
        """

        self.api_client = self.__get_api_client_by_env(config=config)

    def __get_api_client_by_env(self, config: Config):
        # return proper client version by env (cloud : v1, on-prem : v2)
        if config.is_v1():
            return AuthClientV1(config=config)
        else:
            return AuthClientV2(config=config)

    def login(self, email, password, verify_ssl: bool = True) -> TokenResponse:
        return self.api_client.login(
            email=email, password=password, verify_ssl=verify_ssl
        )

    def get_user_info(self, access_token, verify_ssl: bool = True) -> UserResponse:
        return self.api_client.get_user_info(
            access_token=access_token, verify_ssl=verify_ssl
        )

    def get_credit(self, access_token, verify_ssl: bool = True) -> int:
        return self.api_client.get_credit(
            access_token=access_token, verify_ssl=verify_ssl
        )

    def reissue_token(
        self, access_token, refresh_token, verify_ssl: bool = True
    ) -> TokenResponse:
        return self.api_client.reissue_token(
            access_token=access_token,
            refresh_token=refresh_token,
            verify_ssl=verify_ssl,
        )


class TokenHandler:
    def __init__(self, email, password, verify_ssl: bool = True) -> None:
        self.tokens = auth_client.login(
            email=email, password=password, verify_ssl=verify_ssl
        )
        self.email = email
        self.password = password
        self.verify_ssl = verify_ssl

    def check_jwt_exp(self):
        payload = jwt.decode(
            self.tokens.access_token, options={"verify_signature": False}
        )
        return datetime.now(pytz.utc).timestamp() <= payload["exp"]

    def validate_token(self):
        if not self.check_jwt_exp():
            try:
                self.tokens = auth_client.reissue_token(
                    self.tokens.access_token, self.tokens.refresh_token, self.verify_ssl
                )
            except Exception:
                auth_client.login(
                    email=self.email, password=self.password, verify_ssl=self.verify_ssl
                )
                logger.info(
                    "The refresh token has expired. the token has been reissued."
                )


auth_client = AuthClient()