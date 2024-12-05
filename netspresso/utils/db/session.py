from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy_utils import create_database, database_exists

DB_URL = "sqlite:///netspresso.db"
engine = create_engine(
    f"{DB_URL}",
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


if not database_exists(engine.url):
    create_database(engine.url)
