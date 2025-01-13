from fastapi import APIRouter

from app.api.v1.endpoints import model, project, system, user, task

api_router = APIRouter()
api_router.include_router(user.router, prefix="/users", tags=["user"])
api_router.include_router(project.router, prefix="/projects", tags=["project"])
api_router.include_router(model.router, prefix="/models", tags=["model"])
api_router.include_router(task.router, prefix="/tasks", tags=["task"])
api_router.include_router(system.router, prefix="/system", tags=["system"])
