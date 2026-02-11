from app.mlops.mlflow_logger import MLflowLogger
from app.config.config import settings
from typing import Dict, Any

# Global variable to store the current RAG run ID
RAG_RUN_ID = None

def log_rag_experiment(run_name: str = "rag_pipeline"):
    """
    Initialise une run MLflow et logge toute la configuration du pipeline RAG.
    """
    global RAG_RUN_ID
    logger = MLflowLogger(experiment_name="RAG_MediAssist")
    
    # Start run
    run = logger.start_run(run_name=run_name)
    RAG_RUN_ID = run.info.run_id

    
    # Define RAG Configuration
    rag_config = {
        # Global
        "project": settings.PROJECT_NAME,
        "environment": settings.ENVIRONMENT,
        
        # Chunking (Hardcoded for now, ideally matched with actual logic)
        "chunk_strategy": "markdown_header + paragraph_split",
        "chunk_max_tokens": 500,
        "chunk_overlap": 80,
        "chunk_token_estimation": "word_based",
        
        # Embedding
        "embedding_provider": "ollama",
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "embedding_dimension": 384, # all-MiniLM-L6-v2 is 384
        "embedding_normalization": True,
        
        # Retrieval
        "retrieval_vector_db": "qdrant",
        "retrieval_distance": "cosine",
        "retrieval_top_k": 10, # default
        "retrieval_reranking": False,
        
        # LLM
        "llm_model": settings.OLLAMA_MODEL,
        "llm_base_url": settings.OLLAMA_BASE_URL,
        "llm_temperature": 0.2,
        "llm_top_p": 0.9,
        "llm_template": "rag_prompt_v1"
    }
    
    # Log all params
    logger.log_rag_config(rag_config)
    
    return logger, run
