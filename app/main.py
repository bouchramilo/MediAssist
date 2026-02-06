from fastapi import FastAPI
from app.config.config import settings
from app.services.chunking import split_documents
from app.services.pdf_loader import load_pdf

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

@app.get("/")
async def root():
    return {"message": "MediAssist API is running", "environment": settings.ENVIRONMENT}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}




@app.get("/test")
async def test():
    try:
        documents = load_pdf()
        chunks = split_documents(documents=documents)
        
        return {
            "status": "success",
            "count": len(chunks),
            "chunks": [
                {
                    "content": chunk.page_content,
                    "metadata": chunk.metadata
                } for chunk in chunks
            ]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)} 