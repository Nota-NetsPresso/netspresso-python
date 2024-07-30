from pathlib import Path

from .netspresso import TAO, NetsPresso, QAIHub

__all__ = ["NetsPresso", "TAO", "QAIHub"]


version = (Path(__file__).parent / "VERSION").read_text().strip()

__version__ = version
