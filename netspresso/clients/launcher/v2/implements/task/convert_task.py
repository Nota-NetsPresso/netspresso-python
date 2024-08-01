from dataclasses import asdict
from enum import Enum

from loguru import logger

from netspresso.clients.launcher.v2.interfaces import TaskInterface
from netspresso.clients.launcher.v2.schemas import (
    AuthorizationHeader,
    RequestConvert,
    ResponseConvertDownloadModelUrlItem,
    ResponseConvertFrameworkOptionItems,
    ResponseConvertOptionItems,
    ResponseConvertStatusItem,
    ResponseConvertTaskItem,
    UploadFile,
)
from netspresso.clients.utils.requester import Requester
from netspresso.enums import LauncherTask


class ConvertTaskAPI(TaskInterface):
    def __init__(self, url):
        self.task_type = LauncherTask.CONVERT.value
        self.base_url = url
        self.task_base_url = f"{self.base_url}/{self.task_type}/tasks"
        self.option_base_url = f"{self.base_url}/{self.task_type}/options"
        self.model_base_url = f"{self.base_url}/{self.task_type}/models"

    @staticmethod
    def custom_asdict_factory(data):
        def convert_value(obj):
            if isinstance(obj, Enum):
                return obj.value
            return obj

        return {k: convert_value(v) for k, v in data}

    def start(
        self,
        request_body: RequestConvert,
        headers: AuthorizationHeader,
        file: UploadFile = None,
    ) -> ResponseConvertTaskItem:
        endpoint = f"{self.task_base_url}"

        logger.info(f"Request_Body: {asdict(request_body)}")
        response = Requester().post_as_form(
            url=endpoint,
            request_body=asdict(request_body, dict_factory=self.custom_asdict_factory),
            headers=headers.to_dict(),
            binary=file.files if file else None,
        )
        return ResponseConvertTaskItem(**response.json())

    def cancel(self, headers: AuthorizationHeader, task_id: str) -> ResponseConvertTaskItem:
        endpoint = f"{self.task_base_url}/{task_id}/cancel"
        response = Requester().post_as_json(url=endpoint, request_body={}, headers=headers.to_dict())
        return ResponseConvertTaskItem(**response.json())

    def read(self, headers: AuthorizationHeader, task_id: str) -> ResponseConvertTaskItem:
        endpoint = f"{self.task_base_url}/{task_id}"
        response = Requester().get(url=endpoint, headers=headers.to_dict())
        return ResponseConvertTaskItem(**response.json())

    def delete(self, headers: AuthorizationHeader, task_id: str) -> ResponseConvertTaskItem:
        endpoint = f"{self.task_base_url}/{task_id}"
        response = Requester().delete(url=endpoint, headers=headers.to_dict())
        return ResponseConvertTaskItem(**response.json())

    def status(self, headers: AuthorizationHeader, task_id: str) -> ResponseConvertStatusItem:
        endpoint = f"{self.task_base_url}/{task_id}"
        response = Requester().get(url=endpoint, headers=headers.to_dict())
        return ResponseConvertStatusItem(**response.json())

    def options(self, headers: AuthorizationHeader) -> ResponseConvertOptionItems:
        endpoint = f"{self.option_base_url}"
        response = Requester().get(url=endpoint, headers=headers.to_dict())
        return ResponseConvertOptionItems(**response.json())

    def options_by_model_framework(
        self, headers: AuthorizationHeader, model_framework: str
    ) -> ResponseConvertFrameworkOptionItems:
        endpoint = f"{self.option_base_url}/frameworks/{model_framework}"
        response = Requester().get(url=endpoint, headers=headers.to_dict())
        return ResponseConvertFrameworkOptionItems(**response.json())

    def option_by_target_framework(
        self, headers: AuthorizationHeader, target_framework: str
    ) -> ResponseConvertOptionItems:
        endpoint = f"{self.option_base_url}/details/{target_framework}"
        response = Requester().get(url=endpoint, headers=headers.to_dict())
        return ResponseConvertOptionItems(**response.json())

    def get_download_url(
        self, headers: AuthorizationHeader, convert_task_uuid: str
    ) -> ResponseConvertDownloadModelUrlItem:
        endpoint = f"{self.model_base_url}/{convert_task_uuid}"
        response = Requester().get(url=endpoint, headers=headers.to_dict())
        return ResponseConvertDownloadModelUrlItem(**response.json())
