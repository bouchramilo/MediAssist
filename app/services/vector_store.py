# app/services/vector_store.py

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http import models
from app.services.embeddings import get_embedding_function
from app.config import settings
from app.utils.logger import AppLogger
from typing import List, Optional, Dict, Tuple
import uuid

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
        QdrantVectorStore.from_documents(
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
    
    return QdrantVectorStore(
        client=QdrantClient(url=settings.QDRANT_URL),
        collection_name=settings.QDRANT_COLLECTION_NAME,
        embedding=embeddings
    )

def search_semantic(query: str, top_k: int = 10, filters: Optional[Dict] = None) -> List[Tuple[Document, float]]:
    """Recherche sémantique avec scores de similarité"""
    vector_store = get_vector_store()
    
    results = vector_store.similarity_search_with_score(
        query=query,
        k=top_k,
        filter=filters
    )
    
    # Les résultats sont déjà sous forme (Document, score)
    return results

def search_keyword(query: str, top_k: int = 10) -> List[Document]:
    """Recherche par mots-clés en utilisant le payload de Qdrant"""
    client = QdrantClient(url=settings.QDRANT_URL)
    
    
    all_points = client.scroll(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        limit=1000,
        with_payload=True
    )[0]
    
    keywords = [word.lower() for word in query.split() if len(word) > 3]
    
    scored_results = []
    for point in all_points:
        content = point.payload.get('page_content', '').lower()
        metadata = point.payload.get('metadata', {})
        
        # Calculer un score basé sur la présence des mots-clés
        score = 0
        for keyword in keywords:
            if keyword in content:
                score += content.count(keyword)
        
        if score > 0:
            doc = Document(
                page_content=point.payload.get('page_content', ''),
                metadata=metadata
            )
            # Normaliser le score
            normalized_score = min(score / len(keywords), 1.0)
            scored_results.append((doc, normalized_score))
    
    # Trier par score et limiter
    scored_results.sort(key=lambda x: x[1], reverse=True)
    return scored_results[:top_k]

def search_hybrid(query: str, top_k: int = 5, alpha: float = 0.7) -> List[Document]:
    """Recherche hybride: combine sémantique et mots-clés avec pondération alpha
    alpha = poids de la recherche sémantique (0-1)
    """
    logger.info(f"Hybrid search: query='{query}', top_k={top_k}, alpha={alpha}")
    
    # Recherche sémantique
    semantic_results = search_semantic(query, top_k=top_k * 2)
    
    logger.info(f"😊😊 semantic_results : {semantic_results}")
    
    # Recherche par mots-clés
    keyword_results = search_keyword(query, top_k=top_k * 2)
    
    logger.info(f"😊😊 keyword_results : {keyword_results}")
    
    # Dictionnaire pour fusionner les résultats
    merged_results = {}
    
    # Ajouter les résultats sémantiques avec poids alpha
    for doc, score in semantic_results:
        doc_id = doc.metadata.get('_id', str(uuid.uuid4()))
        merged_results[doc_id] = {
            'doc': doc,
            'score': score * alpha
        }
    
    # Ajouter les résultats keywords avec poids (1-alpha)
    for doc, score in keyword_results:
        doc_id = doc.metadata.get('_id', str(uuid.uuid4()))
        if doc_id in merged_results:
            merged_results[doc_id]['score'] += score * (1 - alpha)
        else:
            merged_results[doc_id] = {
                'doc': doc,
                'score': score * (1 - alpha)
            }
    
    # Trier par score décroissant
    sorted_results = sorted(
        merged_results.values(),
        key=lambda x: x['score'],
        reverse=True
    )
    
    # Limiter au top_k
    final_docs = [item['doc'] for item in sorted_results[:top_k]]
    logger.info(f"🔎🔎🔎🔎 Hybrid search - final_docs : {final_docs}")
    
    logger.info(f"🔎 Hybrid search returned {len(final_docs)} documents")
    return final_docs