import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Union
from urllib import request

from loguru import logger

from netspresso.exceptions.common import NotSupportedFrameworkException, NotValidInputModelPath

FRAMEWORK_EXTENSION_MAP = {
    "tensorflow_keras": ".h5",
    "pytorch": ".pt",
    "onnx": ".pt",
    "tensorflow_lite": ".tflite",
    "drpai": ".zip",
    "openvino": ".zip",
    "tensorrt": ".trt",
}


class FileHandler:
    """Utility class for file-related operations."""

    @staticmethod
    def check_exists(folder_path: str) -> bool:
        """Check if the file or folder exists.

        Args:
            folder_path (str): The path to the folder.

        Returns:
            bool: True if the file or folder exists, False otherwise
        """
        return Path(folder_path).exists()

    @staticmethod
    def check_input_model_path(input_model_path: str):
        """Check if the input model path is a file.

        Args:
            input_model_path (str): The path to the input model file.

        Raises:
            FileNotFoundError: If the input model path is not a file.
        """

        if not Path(input_model_path).is_file():
            raise NotValidInputModelPath()

    @staticmethod
    def create_folder(
        folder_path: str,
        parents: bool = True,
        exist_ok: bool = True,
        is_folder_check: bool = True,
    ) -> None:
        """Create a folder.

        Args:
            folder_path (str): The path to the folder to be created.
            parents (bool, optional): If True, also create parent directories if they don't exist.
            exist_ok (bool, optional): If False, raise an error if the folder already exists.
            is_folder_check (bool, optional): If True, check if the folder already exists.

        Raises:
            SystemExit: If the folder already exists and `exist_ok` is False.
        """
        if is_folder_check and not FileHandler.check_exists(folder_path=folder_path):
            Path(folder_path).mkdir(parents=parents, exist_ok=exist_ok)
            logger.info(f"The folder has been created. Local Path: {Path(folder_path)}")
        elif is_folder_check:
            sys.exit(f"This folder already exists. Local Path: {Path(folder_path)}")

    @staticmethod
    def create_unique_folder(folder_path: str) -> Path:
        folder_path = Path(folder_path)
        if not folder_path.exists():
            folder_path.mkdir(parents=True)
            logger.info(f"The folder has been created. Local Path: {folder_path.as_posix()}")
        else:
            count = 1
            while True:
                new_folder_path = folder_path.with_name(f"{folder_path.name} ({count})")
                if not new_folder_path.exists():
                    new_folder_path.mkdir(parents=True)
                    folder_path = new_folder_path
                    logger.info(f"The folder has been created. Local Path: {folder_path.as_posix()}")
                    break
                count += 1

        return folder_path

    @staticmethod
    def create_file_path(folder_path: str, name: str, extension: str) -> Union[str, Path]:
        """Create a file path.

        Args:
            folder_path (str): The path to the folder where the file will be located.
            name (str): The name of the file.
            extension (str): The file extension.

        Returns:
            Union[str, Path]: The full path to the created file.
        """
        return Path(folder_path) / (name + extension)

    @staticmethod
    def download_file(url: str, save_path: Union[str, Path]) -> None:
        """Download a file from the given URL and save it to the specified path.

        Args:
            url (str): The URL of the file to be downloaded.
            save_path (Union[str, Path]): The path where the downloaded file will be saved.
        """
        request.urlretrieve(url, save_path)

    @staticmethod
    def get_extension_by_framework(framework: str) -> str:
        """Get the file extension based on the given framework.

        Args:
            framework (str): The framework name.

        Raises:
            KeyError: If the framework is not found in the extension map.

        Returns:
            str: The file extension corresponding to the framework.
        """
        extension = FRAMEWORK_EXTENSION_MAP.get(framework)
        if extension is None:
            available_frameworks = list(FRAMEWORK_EXTENSION_MAP.keys())
            raise NotSupportedFrameworkException(available_frameworks, framework)
        return extension

    @staticmethod
    def get_path_and_extension(folder_path: str, framework: str) -> Tuple[Path, str]:
        """Prepare the model path by creating folders and generating a default model path.

        Args:
            folder_path (str): The base folder path.
            framework (str): The framework name.
            is_folder_check (bool, optional): If True, check if the folder exists before creating.

        Returns:
            Tuple[Path, str]: A tuple containing the default model path (Path) and the file extension (str).
        """
        default_model_path = (Path(folder_path) / f"{Path(folder_path).name}.ext").resolve()
        extension = FileHandler.get_extension_by_framework(framework=framework)

        return default_model_path, extension

    @staticmethod
    def get_default_model_path(folder_path: str) -> Path:
        """Generates the default model file path based on the provided folder path.

        This method constructs a file path for the default model file located in the
        specified folder. The default model file is named using the name of the base
        folder and has a `.ext` extension. The resulting path is resolved to its absolute
        form.

        Args:
            folder_path (str): The base folder path where the model file is located.

        Returns:
            Path: The default model file path as a `Path` object. This path is formed by
                combining the base folder path with the default file name and extension,
                and is resolved to an absolute path.

        Example:
            >>> FileHandler.get_default_model_path('/models/my_model')
            PosixPath('/models/my_model/my_model.ext')

        Note:
            - Ensure that the base folder path provided is valid and accessible.
            - The extension `.ext` should be replaced with the actual file extension needed
            for the model file.
        """
        default_model_path = (Path(folder_path) / f"{Path(folder_path).name}.ext").resolve()

        return default_model_path

    @staticmethod
    def get_extension(framework: str) -> str:
        """Retrieves the file extension associated with the given framework.

        This method looks up the file extension for a specific framework using the
        `FRAMEWORK_EXTENSION_MAP`. If the framework is not found in the map, it raises
        a `KeyError` with a message listing all supported frameworks.

        Args:
            framework (str): The name of the framework for which the file extension is requested.

        Raises:
            KeyError: If the provided framework is not found in the `FRAMEWORK_EXTENSION_MAP`.
                    The error message includes the list of supported frameworks and the
                    entered framework.

        Returns:
            str: The file extension corresponding to the specified framework. This is a string
                representing the file extension (e.g., '.h5', '.pt', etc.).

        Example:
            >>> FileHandler.get_extension('pytorch')
            '.pt'

            >>> FileHandler.get_extension('unknown_framework')
            KeyError: "The framework supports ['tensorflow_keras', 'pytorch', 'onnx', 'tensorflow_lite', 'drpai', 'openvino', 'tensorrt']. The entered framework is unknown_framework."
        """
        extension = FRAMEWORK_EXTENSION_MAP.get(framework)
        if extension is None:
            available_frameworks = list(FRAMEWORK_EXTENSION_MAP.keys())
            raise NotSupportedFrameworkException(available_frameworks, framework)
        return extension

    @staticmethod
    def load_json(file_path: str):
        """Load JSON data from a file.

        Args:
            file_path (str): Path to the JSON file.

        Returns:
            dict: Loaded JSON data.
        """

        with open(file_path, "r") as json_data:
            data = json.load(json_data)
        return data

    @staticmethod
    def save_json(data: Union[Dict, List[Dict]], file_path: str) -> None:
        """Save data to a JSON file.

        Args:
            data (Union[Dict, List[Dict]]): The data to be saved. This can be a dictionary or a list of dictionaries.
            file_path (str): The file path where the JSON file will be saved.

        Returns:
            None: This function does not return any value.

        Notes:
            The data is written to a JSON file with indentation for readability. The file is saved in the specified directory with the given name.
        """

        with open(file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
        logger.info(f"JSON file saved at {file_path}")

    @staticmethod
    def move_and_cleanup_folders(source_folder: str, destination_folder: str):
        """Move files from the source folder to the destination folder and remove the source folder.

        Args:
            source_folder (str): The path to the source folder.
            destination_folder (str): The path to the destination folder.
        """

        source_folder = Path(source_folder)
        destination_folder = Path(destination_folder)

        for file_path in source_folder.iterdir():
            destination_path = destination_folder / file_path.name
            shutil.move(file_path, destination_path)

        source_folder.rmdir()

    @staticmethod
    def remove_folder(folder_path: str) -> None:
        """Remove a folder and its contents.

        Args:
            folder_path (str): Path to the folder.
        """

        folder_path = Path(folder_path)
        shutil.rmtree(folder_path, ignore_errors=True)

    @staticmethod
    def remove_file(file_path: str) -> None:
        """
        Remove a file.

        Args:
            file_path (str): Path to the file.
        """
        file_path = Path(file_path)

        if file_path.exists() and file_path.is_file():
            file_path.unlink()
            logger.info(f"File '{file_path}' has been removed.")
        else:
            logger.info(f"File '{file_path}' not found or is not a file.")

    @staticmethod
    def read_file_bytes(file_path: str) -> bytes:
        """Read the contents of a file and return them as bytes.

        Args:
            file_path (str): The path to the file to be read.

        Returns:
            bytes: The contents of the file as a byte object.
        """
        with open(file_path, "rb") as f:
            file_byte = f.read()
        return file_byte

    @staticmethod
    def unzip(zip_file_path: str, target_path: str) -> None:
        """Unzip a ZIP file and extract its contents to a specified directory.

        Args:
            zip_path (str): The path to the ZIP file that needs to be unzipped.
            target_path (str): The directory where the contents of the ZIP file will be extracted.

        Returns:
            None
        """
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(target_path)

    @staticmethod
    def rename_file(old_file_path: str, new_file_path: str):
        """Rename a file if it exists.

        Args:
            old_file_path (str): The original file path.
            new_file_path (str): The new file path.

        Returns:
            None
        """
        old_file_path = Path(old_file_path)
        new_file_path = Path(new_file_path)

        if old_file_path.exists():
            old_file_path.rename(new_file_path)
            logger.info(f"File '{old_file_path.name}' has been successfully renamed to '{new_file_path.name}'.")
        else:
            logger.info(f"File '{old_file_path.name}' not found.")
