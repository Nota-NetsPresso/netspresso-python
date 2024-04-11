from dataclasses import dataclass

from netspresso.enums import LauncherTask


@dataclass
class RequestModelUploadUrl:
    """ """

    object_name: str
    task: LauncherTask = LauncherTask.CONVERT.value


@dataclass
class RequestUploadModel:
    """ """

    url: str


@dataclass
class RequestValidateModel:
    """ """

    ai_model_id: str
    display_name: str
