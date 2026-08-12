"""Layer 1 database setup. This file imports models only, never API routers."""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models_layer1 import Base

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./roj_guard.db")
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
