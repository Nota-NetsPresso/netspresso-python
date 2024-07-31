import json
from pathlib import Path
from typing import Any, Dict

from netspresso.metadata.common import BaseMetadata


class MetadataHandler:
    @staticmethod
    def save_metadata(data: BaseMetadata, folder_path: str, file_name: str = "metadata") -> None:
        """Save dictionary data to a JSON file.

        Args:
            data (dict): The dictionary data to be saved.
            file_path (str): The path to the JSON file.

        Returns:
            None
        """
        file_path = Path(folder_path) / f"{file_name}.json"

        with open(file_path, "w") as json_file:
            json.dump(data.asdict(), json_file, indent=4)

    @staticmethod
    def load_json(file_path: str) -> Dict[str, Any]:
        """Load JSON data from a file.

        Args:
            file_path (str): The path to the JSON file.

        Returns:
            dict: Loaded dictionary data.
        """
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        return data
