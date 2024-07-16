from dataclasses import asdict

import requests
from requests_toolbelt import MultipartEncoderMonitor
from requests_toolbelt.multipart.encoder import MultipartEncoder
from tqdm import tqdm

from netspresso.clients.compressor.v2.schemas.common import RequestPagination, UploadFile
from netspresso.clients.compressor.v2.schemas.compression import (
    RequestAutomaticCompressionParams,
    RequestAvailableLayers,
    RequestCreateCompression,
    RequestCreateRecommendation,
    RequestUpdateCompression,
    ResponseCompressionItem,
    ResponseCompressionItems,
    ResponseRecommendationItem,
    ResponseSelectMethodItem,
)
from netspresso.clients.compressor.v2.schemas.model import (
    RequestCreateModel,
    RequestUploadModel,
    RequestValidateModel,
    ResponseModelItem,
    ResponseModelItems,
    ResponseModelUrl,
)
from netspresso.clients.config import Config, Module
from netspresso.clients.utils.common import create_multipart_data, create_progress_func, get_headers, progress_callback
from netspresso.clients.utils.requester import Requester


class CompressorAPIClient:
    def __init__(self):
        self.config = Config(Module.COMPRESSOR)
        self.host = self.config.HOST
        self.port = self.config.PORT
        self.prefix = self.config.URI_PREFIX
        self.url = f"{self.host}:{self.port}{self.prefix}"

    def is_cloud(self) -> bool:
        # TODO
        return self.config.is_cloud()

    def create_model(
        self, request_data: RequestCreateModel, access_token: str, verify_ssl: bool = True
    ) -> ResponseModelUrl:
        url = f"{self.url}/models"
        response = Requester.post_as_json(url=url, request_body=asdict(request_data), headers=get_headers(access_token))

        return ResponseModelUrl(**response.json())

    def upload_model(
        self, request_data: RequestUploadModel, file: UploadFile, access_token: str, verify_ssl: bool = True
    ) -> bool:
        url = f"{self.url}/models/upload"

        file_info = file.files[0][1]

        multipart_data = create_multipart_data(request_data.url, file_info)
        progress = create_progress_func(multipart_data)

        # Wrap the encoder with MultipartEncoderMonitor
        monitor = MultipartEncoderMonitor(multipart_data, lambda monitor: progress_callback(monitor, progress))

        headers = get_headers(access_token)
        headers["Content-Type"] = monitor.content_type

        response = requests.post(url=url, data=monitor, headers=headers, verify=verify_ssl)

        return response.text

    def validate_model(
        self, ai_model_id: str, request_data: RequestValidateModel, access_token: str, verify_ssl: bool = True
    ) -> ResponseModelItem:
        url = f"{self.url}/models/{ai_model_id}/validate"
        response = Requester.post_as_json(
            url=url,
            request_body=asdict(request_data),
            headers=get_headers(access_token),
        )

        return ResponseModelItem(**response.json())

    def read_models(
        self, request_params: RequestPagination, access_token: str, verify_ssl: bool = True
    ) -> ResponseModelItems:
        url = f"{self.url}/models"
        response = Requester.get(
            url=url,
            params=asdict(request_params),
            headers=get_headers(access_token),
        )

        return ResponseModelItems(**response.json())

    def read_model(self, ai_model_id: str, access_token: str, verify_ssl: bool = True) -> ResponseModelItem:
        url = f"{self.url}/models/{ai_model_id}"
        response = Requester.get(
            url=url,
            headers=get_headers(access_token),
        )

        return ResponseModelItem(**response.json())

    def download_model(self, ai_model_id: str, access_token: str, verify_ssl: bool = True) -> ResponseModelUrl:
        url = f"{self.url}/models/{ai_model_id}/download"
        response = Requester.post_as_json(
            url=url,
            headers=get_headers(access_token),
        )

        return ResponseModelUrl(**response.json())

    def create_compression(
        self, request_data: RequestCreateCompression, access_token: str, verify_ssl: bool = True
    ) -> ResponseCompressionItem:
        url = f"{self.url}/compressions"
        response = Requester.post_as_json(
            url=url,
            request_body=asdict(request_data),
            headers=get_headers(access_token),
        )

        return ResponseCompressionItem(**response.json())

    def read_compressions(
        self, request_params: RequestPagination, access_token: str, verify_ssl: bool = True
    ) -> ResponseCompressionItems:
        url = f"{self.url}/compressions"
        response = Requester.get(
            url=url,
            params=asdict(request_params),
            headers=get_headers(access_token),
        )

        return ResponseCompressionItems(**response.json())

    def read_compression(
        self, compression_id: str, access_token: str, verify_ssl: bool = True
    ) -> ResponseCompressionItem:
        url = f"{self.url}/compressions/{compression_id}"
        response = Requester.get(
            url=url,
            headers=get_headers(access_token),
        )

        return ResponseCompressionItem(**response.json())

    def create_recommendation(
        self, compression_id: str, request_data: RequestCreateRecommendation, access_token: str, verify_ssl: bool = True
    ) -> ResponseRecommendationItem:
        url = f"{self.url}/compressions/{compression_id}/recommendation"
        response = Requester.post_as_json(
            url=url,
            request_body=asdict(request_data),
            headers=get_headers(access_token),
        )

        return ResponseRecommendationItem(**response.json())

    def compress_model(
        self, compression_id: str, request_data: RequestUpdateCompression, access_token: str, verify_ssl: bool = True
    ) -> ResponseCompressionItem:
        url = f"{self.url}/compressions/{compression_id}"
        response = Requester.put(
            url=url,
            request_body=asdict(request_data),
            headers=get_headers(access_token),
        )

        return ResponseCompressionItem(**response.json())

    def compress_model_with_automatic(
        self,
        ai_model_id: str,
        request_data: RequestAutomaticCompressionParams,
        access_token: str,
        verify_ssl: bool = True,
    ) -> ResponseCompressionItem:
        url = f"{self.url}/models/{ai_model_id}/auto_compress"
        response = Requester.post_as_json(
            url=url,
            request_body=asdict(request_data),
            headers=get_headers(access_token),
        )

        return ResponseCompressionItem(**response.json())

    def get_available_layers(
        self, ai_model_id: str, request_data: RequestAvailableLayers, access_token: str, verify_ssl: bool = True
    ) -> ResponseSelectMethodItem:
        url = f"{self.url}/models/{ai_model_id}/available_layers"
        response = Requester.post_as_json(
            url=url,
            request_body=asdict(request_data),
            headers=get_headers(access_token),
        )

        return ResponseSelectMethodItem(**response.json())

    def upload_dataset(self, compression_id: str, file: UploadFile, access_token: str, verify_ssl: bool = True):
        url = f"{self.url}/compressions/{compression_id}/datasets"
        response = Requester.post_as_form(
            url=url,
            binary=file.files,
            headers=get_headers(access_token),
        )

        return ResponseCompressionItem(**response.json())


compressor_client_v2 = CompressorAPIClient()
