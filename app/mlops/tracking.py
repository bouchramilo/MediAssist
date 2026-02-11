from typing import Optional, Tuple, Dict, Any
import mlflow
from app.mlops.mlflow_logger import MLflowLogger
from app.config.config import settings
from app.services.chunking import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

# Global variables
RAG_RUN_ID: Optional[str] = None
_logger_instance: Optional[MLflowLogger] = None

def get_current_logger() -> Optional[MLflowLogger]:
    return _logger_instance

def log_rag_experiment(run_name: str = "rag_pipeline") -> Tuple[MLflowLogger, mlflow.ActiveRun]:
    """
    Initialise une run MLflow et logge toute la configuration du pipeline RAG.
    Utilise les paramètres définis dans chunking et config.
    """
    global RAG_RUN_ID, _logger_instance
    
    if _logger_instance and RAG_RUN_ID:
        # If already initialized, return existing
        return _logger_instance, mlflow.active_run()

    _logger_instance = MLflowLogger(experiment_name="RAG_MediAssist")
    
    # Start run
    run = _logger_instance.start_run(run_name=run_name)
    RAG_RUN_ID = run.info.run_id

    
    # Define RAG Configuration dynamically
    rag_config = {
        # Global
        "project": settings.PROJECT_NAME,
        "environment": settings.ENVIRONMENT,
        
        # Chunking
        "chunk_strategy": "markdown_header + paragraph_split",
        "chunk_max_tokens": DEFAULT_CHUNK_SIZE,
        "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
        "chunk_token_estimation": "word_based",
        
        # Embedding
        "embedding_provider": "ollama",
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "embedding_dimension": 384, # Assumed for connection check, strictly ideally dynamic but acceptable here
        "embedding_normalization": True,
        
        # Retrieval
        "retrieval_vector_db": "qdrant",
        "retrieval_distance": "cosine",
        "retrieval_top_k": 10,
        "retrieval_reranking": False,
        
        # LLM
        "llm_model": settings.OLLAMA_MODEL,
        "llm_base_url": settings.OLLAMA_BASE_URL,
        "llm_temperature": 0.2,
        "llm_top_p": 0.9,
        "llm_template": "rag_prompt_v1"
    }
    
    # Log all params
    _logger_instance.log_rag_config(rag_config)
    
    return _logger_instance, run
