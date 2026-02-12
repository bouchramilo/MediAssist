# llm.py

from langchain_ollama import ChatOllama
from app.config.config import settings
from app.utils.logger import AppLogger


logger = AppLogger.get_logger(__name__)

def create_llm() -> ChatOllama:
    """
    Crée et configure une instance de ChatOllama.
    Log la configuration dans MLflow si un run est actif.
    """
    logger.info(f"Connecting to Ollama LLM ({settings.OLLAMA_MODEL}) at {settings.OLLAMA_BASE_URL}...")

    # MLflow logging


    try:
        llm = ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=settings.LLM_TEMPERATURE,
            num_predict=settings.LLM_NUM_PREDICT,
            top_p=settings.LLM_TOP_P,
            repeat_penalty=settings.LLM_REPEAT_PENALTY,
        )
        logger.info("Ollama LLM connected successfully.")
        return llm

    except Exception as e:
        logger.error(f"Failed to connect to Ollama LLM: {e}")
        raise
