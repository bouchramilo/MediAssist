from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http import models
from app.services.embeddings import get_embedding_function
from app.config import settings
from app.utils.logger import AppLogger
from typing import List, Optional, Dict

try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.schema import Document
    except ImportError:
        from langchain.docstore.document import Document

logger = AppLogger.get_logger(__name__)

def create_qdrant_collection():
    """Crée la collection Qdrant si elle n'existe pas"""
    client = QdrantClient(url=settings.QDRANT_URL)
    
    collections = client.get_collections()
    if settings.QDRANT_COLLECTION_NAME in [c.name for c in collections.collections]:
        logger.info(f"Collection '{settings.QDRANT_COLLECTION_NAME}' existe déjà")
        return True
    
    embeddings = get_embedding_function()
    test_embedding = embeddings.embed_query("test")
    
    client.create_collection(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(
            size=len(test_embedding),
            distance=Distance.COSINE
        )
    )
    
    logger.info(f"Collection '{settings.QDRANT_COLLECTION_NAME}' créée")
    return True

def store_embeddings(chunks: List[Document]):
    """Stocke les embeddings dans Qdrant"""
    try:
        create_qdrant_collection()
        
        embeddings = get_embedding_function()
        
        # Stocker avec LangChain
        Qdrant.from_documents(
            documents=chunks,
            embedding=embeddings,
            url=settings.QDRANT_URL,
            collection_name=settings.QDRANT_COLLECTION_NAME,
            force_recreate=False
        )
        
        logger.info(f"{len(chunks)} documents stockés")
        return True
        
    except Exception as e:
        logger.error(f"Erreur stockage: {e}")
        raise

def get_vector_store():
    """Récupère le vector store pour la recherche"""
    embeddings = get_embedding_function()
    
    return Qdrant(
        client=QdrantClient(url=settings.QDRANT_URL),
        collection_name=settings.QDRANT_COLLECTION_NAME,
        embeddings=embeddings
    )

def search_semantic(query: str, top_k: int = 10, filters: Optional[Dict] = None):
    """Recherche sémantique (par similarité vectorielle)"""
    vector_store = get_vector_store()
    
    return vector_store.similarity_search_with_score(
        query=query,
        k=top_k,
        filter=filters
    )

def search_keyword(query: str, filters: Optional[Dict] = None, top_k: int = 10):
    """Recherche par mots-clés (via filtres Qdrant)"""
    keywords = [word.lower() for word in query.split() if len(word) > 3]
    
    if not keywords:
        return []
    
    filter_condition = models.Filter(
        should=[
            models.FieldCondition(
                key="metadata.keywords",
                match=models.MatchValue(value=keyword)
            )
            for keyword in keywords[:5]
        ]
    )
    
    if filters:
        pass
    
    client = QdrantClient(url=settings.QDRANT_URL)
    results = client.scroll(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        scroll_filter=filter_condition,
        limit=top_k,
        with_payload=True
    )
    
    formatted_results = []
    for point in results[0]:
        formatted_results.append({
            'content': point.payload.get('page_content', ''),
            'metadata': point.payload.get('metadata', {}),
            'score': 0.5 
        })
    
    return formatted_results

def search_hybrid(query: str, top_k: int = 10, alpha: float = 0.7):
    """Recherche hybride: combine sémantique et mots-clés"""
    # 1. Recherche sémantique
    semantic_results = search_semantic(query, top_k=top_k*2)
    
    # 2. Recherche par mots-clés
    keyword_results = search_keyword(query, top_k=top_k*2)
    
    # 3. Fusion simple 
    all_results = []
    
    # Ajouter les résultats sémantiques
    for doc, score in semantic_results:
        all_results.append({
            'doc': doc,
            'score': score * alpha,
            'type': 'semantic'
        })
    
    # Ajouter les résultats mots-clés 
    for result in keyword_results:
        all_results.append({
            'doc': result,
            'score': result['score'] * (1 - alpha),
            'type': 'keyword'
        })
    
    # Trier par score et retourner les top_k
    all_results.sort(key=lambda x: x['score'], reverse=True)
    
    return [result['doc'] for result in all_results[:top_k]]