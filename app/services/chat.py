from app.services.rag_pipeline import initialize_rag_system
from app.utils.logger import AppLogger

logger = AppLogger.get_logger(__name__)

_qa_chain = None

def get_qa_chain():
    global _qa_chain
    if _qa_chain is None:
        logger.info("Initializing RAG chain for the first time...")
        _qa_chain = initialize_rag_system(force_recreate_db=False)
    return _qa_chain

async def ask_question(question: str):
    try:
        chain = get_qa_chain()
        res = chain.invoke(question)
        
        answer = res["answer"]
        source_docs = res["context"]
        
        sources = [doc.metadata.get("source", "unknown") for doc in source_docs]
        
        sources = list(set(sources))
        
        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        logger.error(f"Error in ask_question: {e}")
        return {
            "answer": f"Une erreur est survenue lors du traitement de votre demande : {e}",
            "sources": []
        }
