from app.utils.logger import AppLogger
from langchain_ollama import OllamaLLM
from app.config.config import settings

logger = AppLogger.get_logger(__name__)

def create_llm() -> OllamaLLM:
    """
    Crée et configure une instance de OllamaLLM.
    """
    logger.info(f"Connecting to Ollama LLM ({settings.OLLAMA_MODEL}) at {settings.OLLAMA_BASE_URL}...")
    
    try:
        llm = OllamaLLM(
            model=settings.OLLAMA_MODEL,  # "llama3.1"
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0.2,  # Bas pour des réponses précises
            num_predict=1024,  # Taille max des réponses
            num_ctx=4096,  # Taille du contexte
            top_p=0.9,  # Pour éviter les réponses farfelues
            repeat_penalty=1.1,  # Évite la répétition
        )
        return llm
    except Exception as e:
        logger.error(f"Failed to connect to Ollama LLM: {e}")
        raise