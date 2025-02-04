from pathlib import Path

from .netspresso import NPQAI, TAO, NetsPresso

__all__ = ["NetsPresso", "TAO", "NPQAI"]


version = (Path(__file__).parent / "VERSION").read_text().strip()

__version__ = version
