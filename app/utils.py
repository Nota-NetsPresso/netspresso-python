import hashlib

from nanoid import generate


def generate_id(entity: str, size: int = 10) -> str:
    nano_id = generate(size=size)
    return f"{entity}_{nano_id}"


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()
