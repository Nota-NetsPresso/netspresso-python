from dataclasses import dataclass


@dataclass
class LoginRequest:
    username: str
    password: str
