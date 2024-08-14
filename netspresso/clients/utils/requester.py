from typing import Optional

import requests
from requests import Response

from netspresso.exceptions.common import GatewayTimeoutException, UnexpetedException


class Requester:
    @staticmethod
    def __make_response(response: Response) -> Response:
        if response.ok:
            return response
        else:
            try:
                error_message = response.json()
                raise Exception(error_message)
            except ValueError:
                if response.status_code == 504:
                    raise GatewayTimeoutException(error_log=response.text) from None
                else:
                    raise UnexpetedException(error_log=response.text, status_code=response.status_code) from None


    @staticmethod
    def get(url: str, params: Optional[dict] = None, headers=None, **kwargs) -> Response:
        response = requests.get(url, headers=headers, params=params, **kwargs)

        return Requester.__make_response(response=response)

    @staticmethod
    def post_as_form(url: str, request_body: Optional[dict] = None, binary=None, headers=None, **kwargs) -> Response:
        response = requests.post(url, headers=headers, data=request_body, files=binary, **kwargs)

        return Requester.__make_response(response=response)

    @staticmethod
    def post_as_json(url: str, request_body: dict = None, headers=None, **kwargs) -> Response:
        response = requests.post(url, headers=headers, json=request_body, **kwargs)

        return Requester.__make_response(response=response)

    @staticmethod
    def put(url: str, request_body: dict, headers=None, **kwargs) -> Response:
        response = requests.put(url, headers=headers, json=request_body, **kwargs)

        return Requester.__make_response(response=response)

    @staticmethod
    def patch(url: str, request_body: dict, headers=None, **kwargs) -> Response:
        response = requests.patch(url, headers=headers, json=request_body, **kwargs)

        return Requester.__make_response(response=response)

    @staticmethod
    def delete(url: str, headers=None, **kwargs) -> Response:
        response = requests.delete(url, headers=headers, **kwargs)

        return Requester.__make_response(response=response)
