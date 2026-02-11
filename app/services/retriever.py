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

            # MLflow logging
            if mlflow.active_run():
                mlflow.log_params({
                    "retriever_type": "hybrid",
                    "retriever_top_k": self.top_k,
                    "retriever_alpha": self.alpha,
                    "retriever_framework": "langchain",
                    "retriever_reranking": "none"
                })

                mlflow.log_metrics({
                    "retrieved_documents_count": len(results)
                })

                # Log de la requête
                mlflow.log_text(query, "retrieval_query.txt")

            return results

        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {e}")

            if mlflow.active_run():
                mlflow.set_tag("retriever_status", "failed")

            return []

def create_retriever(top_k: int = 5, alpha: float = 0.7) -> HybridRetriever:
    """
    Crée et retourne une instance de HybridRetriever.
    """
    logger.info(f"Creating HybridRetriever with top_k={top_k}, alpha={alpha}")
    
    if mlflow.active_run():
        mlflow.log_params({
            "retriever_init_top_k": top_k,
            "retriever_init_alpha": alpha
        })
        mlflow.set_tag("rag_component", "retriever")
        
    return HybridRetriever(top_k=top_k, alpha=alpha)
