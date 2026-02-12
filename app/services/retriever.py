# app/services/retriever.py - Déjà bon
from typing import List, Optional
try:
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
except ImportError:
    from langchain.callbacks.manager import CallbackManagerForRetrieverRun
    from langchain.schema.retriever import BaseRetriever
    from langchain.schema import Document

import mlflow
from app.services.vector_store import search_hybrid
from app.utils.logger import AppLogger

logger = AppLogger.get_logger(__name__)

class HybridRetriever(BaseRetriever):
    """
    Retriever hybride combinant recherche sémantique et par mots-clés.
    """
    top_k: int = 5
    alpha: float = 0.7

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        try:
            logger.info(f"Hybrid search for query: {query}")

            results = search_hybrid(
                query=query,
                top_k=self.top_k,
                alpha=self.alpha
            )

            return results

        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {e}")
            return []

def create_retriever(top_k: int = 5, alpha: float = 0.7) -> HybridRetriever:
    """
    Crée et retourne une instance de HybridRetriever.
    """
    logger.info(f"Creating HybridRetriever with top_k={top_k}, alpha={alpha}")
    return HybridRetriever(top_k=top_k, alpha=alpha)