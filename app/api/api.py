from fastapi import APIRouter

from app.api.v1.endpoints import project

api_router = APIRouter()
api_router.include_router(project.router, prefix="/projects", tags=["project"])
