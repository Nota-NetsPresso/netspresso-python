from typing import List

from fastapi import FastAPI
from fastapi.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from app.api.api import api_router
from app.configs.settings import settings


def init_routers(app: FastAPI) -> None:
    app.include_router(api_router, prefix=settings.API_PREFIX)


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

    return app


app = create_app()
