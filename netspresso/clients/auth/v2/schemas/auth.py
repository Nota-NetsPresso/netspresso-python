from dataclasses import dataclass


@dataclass
class TokenRefreshRequest:
    refresh_token: str
