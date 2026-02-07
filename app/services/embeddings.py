# Génère les embeddings
# Utilise HuggingFace ou Ollama
from functools import lru_cache
from langchain_community.embeddings import OllamaEmbeddings

from app.utils.logger import AppLogger

logger = AppLogger.get_logger(__name__)

@lru_cache()
def get_embedding_function(model_name: str = "nomic-embed-text"):
    logger.info(f"Loading embedding model: {model_name} via Ollama")
    
    from app.config import settings
    
    try:
        embeddings = OllamaEmbeddings(
            model=model_name,
            base_url=settings.OLLAMA_BASE_URL
        )
        return embeddings
    except Exception as e:
        logger.error(f"Fialed te load embedding model: {e}")
        raise e