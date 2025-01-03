from contextlib import contextmanager
import os
from typing import Generator

from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy_utils import create_database, database_exists

DB_USER: str = os.environ.get("DB_USER")
DB_PASSWORD: str = os.environ.get("DB_PASSWORD")
DB_ADDRESS: str = os.environ.get("DB_ADDRESS")
DB_PORT: int = int(os.environ.get("DB_PORT"))
DB_NAME: str = os.environ.get("DB_NAME")
DB_URI: str = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_ADDRESS}:{DB_PORT}/{DB_NAME}"

engine = create_engine(
    DB_URI,
    pool_pre_ping=True,
    pool_use_lifo=True,
    pool_recycle=3600,
)
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False,
)

Base = declarative_base()


def get_db() -> Generator:
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()


@contextmanager
def get_db_session():
    db = None
    try:
        db = SessionLocal()
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        raise
    finally:
        if db:
            db.close()


def check_database(engine):
    if not database_exists(engine.url):
        logger.info("The database did not exist, so it has been created.")
        create_database(engine.url)
    else:
        logger.info("The database has already been created.")

check_database(engine=engine)
