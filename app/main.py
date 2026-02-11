from fastapi import FastAPI
from app.config.config import settings
from app.services.chat import ask_question
from app.api import user, admin, chat, documents
from app.config.database import init_db
from app.services.vector_store import create_qdrant_collection

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    try:
        create_qdrant_collection()
    except Exception as e:
        print(f"Error initializing Qdrant: {e}")
        
    try:
        from app.mlops.tracking import log_rag_experiment
        log_rag_experiment()
        print("RAG Logging initialized.")
    except Exception as e:
        print(f"Error initializing RAG logging: {e}")
        
    yield

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

@app.get("/")
async def root():
    return {"message": "MediAssist API is running", "environment": settings.ENVIRONMENT}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}



app.include_router(user.router, prefix=settings.API_V1_STR)
app.include_router(admin.router, prefix=settings.API_V1_STR)
app.include_router(chat.router, prefix=settings.API_V1_STR)
app.include_router(documents.router, prefix=settings.API_V1_STR)





