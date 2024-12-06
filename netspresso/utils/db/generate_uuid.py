from nanoid import generate


def generate_uuid(entity: str, size: int = 10) -> str:
    nano_id = generate(size=size)
    return f"{entity}_{nano_id}"
