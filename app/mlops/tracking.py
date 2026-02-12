# tracking.py
from typing import Optional, Tuple, Dict, Any
import mlflow
from app.mlops.mlflow_logger import MLflowLogger
from app.config.config import settings
from app.services.chunking import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

def create_query_run(run_name_prefix: str = "query") -> Tuple[MLflowLogger, mlflow.ActiveRun]:
    """
    Initialise une NOUVELLE run MLflow pour une requête spécifique.
    Logge toute la configuration du pipeline RAG pour cette run.
    """
    logger_instance = MLflowLogger(experiment_name="RAG_MediAssist")
    
    run = logger_instance.start_run(run_name=run_name_prefix)
    
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
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "embedding_dimension": settings.EMBEDDING_DIMENSION,
        
        # Retrieval
        "retrieval_vector_db": "qdrant",
        "retrieval_distance": "cosine",
        "retrieval_top_k": settings.RETRIEVAL_TOP_K,
        "retrieval_reranking": settings.RETRIEVAL_RERANKING,
        
        # LLM
        "llm_model": settings.OLLAMA_MODEL,
        "llm_base_url": settings.OLLAMA_BASE_URL,
        "llm_temperature": settings.LLM_TEMPERATURE,
        "llm_top_p": settings.LLM_TOP_P,
        "llm_repeat_penalty": settings.LLM_REPEAT_PENALTY,
        "llm_num_predict": settings.LLM_NUM_PREDICT,
        "llm_template": "rag_prompt_v1"
    }
    
    # Log all params for THIS run
    logger_instance.log_rag_config(rag_config)
    
    return logger_instance, run
