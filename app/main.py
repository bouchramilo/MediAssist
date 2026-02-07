from fastapi import FastAPI
from app.config.config import settings
from app.services.chunking import split_documents
from app.services.pdf_loader import load_pdf
from app.services.llm import create_llm
from app.services.chat import ask_question

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




@app.get("/chunks")
async def get_chunks():
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


@app.get("/documents")
async def get_documents(limit: int = 10):
    try:
        documents = load_pdf()
        
        limited_docs = documents[:limit]
        
        return {
            "status": "success",
            "total_count": len(documents),
            "returned_count": len(limited_docs),
            "documents": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in limited_docs
            ]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)} 


@app.get("/llmmodel")
async def get_documents():
    try:
        model = create_llm()
        
        
        return {
            "status": "success",
            "model": model,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)} 


@app.get("/query")
async def get_documents(question:str = "Hello , can you help me to fix a machine medical?"):
    try:
        response = await ask_question(question)
        
        
        return {
            "status": "success",
            "response": response,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)} 