from dataclasses import asdict

from netspresso.clients.compressor.v2.schemas.common import RequestPagination, UploadFile
from netspresso.clients.compressor.v2.schemas.compression import (
    RequestAutomaticCompressionParams,
    RequestAvailableLayers,
    RequestCreateCompression,
    RequestCreateRecommendation,
    RequestUpdateCompression,
    ResponseAvailableLayersItem,
    ResponseCompressionItem,
    ResponseCompressionItems,
    ResponseRecommendationItem,
)
from netspresso.clients.compressor.v2.schemas.model import (
    RequestCreateModel,
    RequestUploadModel,
    RequestValidateModel,
    ResponseModelItem,
    ResponseModelItems,
    ResponseModelUploadUrl,
)
from netspresso.clients.config import Config, Module
from netspresso.clients.utils.common import get_headers
from netspresso.clients.utils.requester import Requester


class CompressorAPIClient:
    def __init__(self):
        self.config = Config(Module.COMPRESSOR)
        self.host = self.config.HOST
        self.port = self.config.PORT
        self.prefix = self.config.URI_PREFIX
        self.url = f"{self.host}:{self.port}{self.prefix}"

    def create_model(
        self, request_data: RequestCreateModel, access_token: str, verify_ssl: bool = True
    ) -> ResponseModelUploadUrl:
        url = f"{self.url}/models"
        response = Requester.post_as_json(url=url, request_body=asdict(request_data), headers=get_headers(access_token))

        return ResponseModelUploadUrl(**response.json())

    def upload_model(self, request_data: RequestUploadModel, file: UploadFile, access_token: str) -> bool:
        url = f"{self.url}/models/upload"
        response = Requester.post_as_form(
            url=url,
            binary=file.files,
            request_body=asdict(request_data),
            headers=get_headers(access_token),
        )

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

        return ResponseModelItems(**response.json())

    def read_compression(
        self, compression_id: str, access_token: str, verify_ssl: bool = True
    ) -> ResponseCompressionItem:
        url = f"{self.url}/compressions/{compression_id}"
        response = Requester.get(
            url=url,
            headers=get_headers(access_token),
        )

        return ResponseModelItem(**response.json())

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
    ) -> ResponseModelItem:
        url = f"{self.url}/compressions/{compression_id}"
        response = Requester.post_as_json(
            url=url,
            request_body=asdict(request_data),
            headers=get_headers(access_token),
        )

        return ResponseModelItem(response.json())

    def auto_compress(
        self,
        ai_model_id: str,
        request_data: RequestAutomaticCompressionParams,
        access_token: str,
        verify_ssl: bool = True,
    ) -> ResponseModelItem:
        url = f"{self.url}/models/{ai_model_id}/auto_compress"
        response = Requester.post_as_json(
            url=url,
            request_body=asdict(request_data),
            headers=get_headers(access_token),
        )

        return ResponseModelItem(response.json())

    def get_available_layers(
        self, ai_model_id: str, request_data: RequestAvailableLayers, access_token: str, verify_ssl: bool = True
    ) -> ResponseAvailableLayersItem:
        url = f"{self.url}/models/{ai_model_id}/available_layers"
        response = Requester.post_as_json(
            url=url,
            request_body=asdict(request_data),
            headers=get_headers(access_token),
        )

        return ResponseAvailableLayersItem(**response.json())


compressor_client_v2 = CompressorAPIClient()