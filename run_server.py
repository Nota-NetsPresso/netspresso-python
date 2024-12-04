import uvicorn

from app.configs.settings import settings
from app.main import app

if __name__ == "__main__":
    uvicorn.run(
        app=app,
        host=settings.SERVER_ADDRESS,
        port=settings.SERVER_PORT,
        workers=settings.SERVER_WORKERS,
    )
