from app.services.rag_pipeline import initialize_rag_system
from app.utils.logger import AppLogger
from app.mlops.evaluation import evaluate_rag
from app.mlops.mlflow_logger import MLflowLogger
from app.mlops import tracking
import mlflow

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
        
        # MLOps: Evaluation & Logging
        try:
            logger.info(f"DEBUG: Attempting MLOps logging. RAG_RUN_ID={tracking.RAG_RUN_ID}")
            if tracking.RAG_RUN_ID:
                chunk_texts = [doc.page_content for doc in source_docs]
                
                # Evaluate
                logger.info("DEBUG: Starting DeepEval evaluation...")
                metrics = evaluate_rag(
                    query=question,
                    response=answer,
                    context=chunk_texts
                )
                logger.info(f"DEBUG: Evaluation complete. Metrics: {metrics}")
                
                # Log to MLflow using the persistent run ID
                mlflow_logger = MLflowLogger(run_id=tracking.RAG_RUN_ID)
                mlflow_logger.log_metrics(metrics)
                logger.info("DEBUG: Metrics logged to MLflow.")
                
                # Log conversation pair
                mlflow_logger.log_text(f"Q: {question}\nA: {answer}", f"conversation_{len(question)}.txt")
            else:
                logger.warning("No active RAG run found (tracking.RAG_RUN_ID is None). Skipping logging.")
                
        except Exception as e:
            logger.warning(f"MLOps Evaluation failed: {e}")
            import traceback
            logger.warning(traceback.format_exc())

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

