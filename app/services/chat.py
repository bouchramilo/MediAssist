from app.services.rag_pipeline import initialize_rag_system
from app.utils.logger import AppLogger
from app.mlops.evaluation import evaluate_rag

from app.mlops import tracking
import mlflow
from datetime import datetime

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
        query_run = None
        mlflow_logger = None
        try:
            # Create a dedicated run for this query
            mlflow_logger, query_run = tracking.create_query_run(run_name_prefix=f"query_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
            run_id = query_run.info.run_id
            
            logger.info(f"DEBUG: MLOps Run started. Run ID: {run_id}")
            
            chunk_texts = [doc.page_content for doc in source_docs]
            
            # Evaluate
            logger.info("DEBUG: Starting DeepEval evaluation...")
            metrics = evaluate_rag(
                query=question,
                response=answer,
                context=chunk_texts
            )
            logger.info(f"DEBUG: Evaluation complete. Metrics: {metrics}")
            
            # Log to MLflow
            print(f"DEBUG: Logging metrics to MLflow run {run_id}: {metrics}")
            mlflow_logger.log_metrics(metrics)
            
            # Log conversation pair
            artifact_filename = "conversation.txt"
            print(f"DEBUG: Logging conversation artifact: {artifact_filename}")
            mlflow_logger.log_text(f"Q: {question}\nA: {answer}", artifact_filename)
                
        except Exception as e:
            logger.warning(f"MLOps Evaluation failed: {e}")
            import traceback
            logger.warning(traceback.format_exc())
            
        finally:
            # Ensure run is ended
            if mlflow_logger:
                mlflow_logger.end_run()
                logger.info("DEBUG: MLOps Run ended.")

                

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

