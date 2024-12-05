import bcrypt
from nanoid import generate


def generate_id(entity: str, size: int = 10) -> str:
    nano_id = generate(size=size)
    return f"{entity}_{nano_id}"


def hash_password(password: str) -> str:
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    return hashed_password.decode('utf-8')
