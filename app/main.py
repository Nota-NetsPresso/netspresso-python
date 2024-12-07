from typing import List

from fastapi import FastAPI, Request, status
from fastapi.middleware import Middleware
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

from app.api.api import api_router
from app.configs.settings import settings
from netspresso.exceptions.common import PyNPException
from netspresso.exceptions.status import STATUS_MAP


def init_routers(app: FastAPI) -> None:
    app.include_router(api_router, prefix=settings.API_PREFIX)


def init_exceptions(app: FastAPI) -> None:
    @app.exception_handler(PyNPException)
    async def http_exception_handler(request: Request, exc: PyNPException):
        status_code = STATUS_MAP.get(exc.detail["error_code"], status.HTTP_500_INTERNAL_SERVER_ERROR)

        return JSONResponse(status_code=status_code, content=exc.detail)

def make_middleware() -> List[Middleware]:
    origins = ["*"]
    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["Access-Token", "Authorization", "Content-Deposition"],
        ),
    ]

    return middleware


def create_app():
    app = FastAPI(
        title="NetsPresso Backend for NP GUI",
        openapi_url=f"{settings.API_PREFIX}/openapi.json",
        version="0.0.1",
        middleware=make_middleware(),
    )
    init_routers(app=app)
    init_exceptions(app=app)

    return app


app = create_app()
