from fastapi import APIRouter
from app.services.chunking import split_documents
from app.services.pdf_loader import load_pdf
from app.services.llm import create_llm

router = APIRouter(tags=["Documents"])

@router.get("/chunks")
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

@router.get("/documents")
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

@router.get("/llmmodel")
async def get_llm_model():
    try:
        model = create_llm()
        
        return {
            "status": "success",
            "model": model,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)} 
