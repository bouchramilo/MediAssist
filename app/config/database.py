# app/config/config.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from app.config import settings

engine = create_engine(
    settings.DATABASE_URL,
    echo=True,
    future=True
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    future=True
)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    from app import models
    try:
        Base.metadata.create_all(bind=engine)
        print("Tables créées avec succès !")
    except SQLAlchemyError as e:
        print(f"Erreur lors de la création des tables : {e}")
