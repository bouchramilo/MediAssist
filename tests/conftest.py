import pytest
from unittest.mock import Mock, patch
from app.config import settings

@pytest.fixture
def mock_docs():
    """Fixture retournant une liste de documents mockés."""
    doc1 = Mock()
    doc1.page_content = "Contenu de la page 1"
    doc1.metadata = {"source": "pdf1"}
    doc2 = Mock()
    doc2.page_content = "Contenu de la page 2"
    doc2.metadata = {"source": "pdf1"}
    return [doc1, doc2]

@pytest.fixture
def mock_qdrant_client():
    """Fixture mockant le client Qdrant dans le module vector_store."""
    with patch('app.services.vector_store.QdrantClient') as mock:
        yield mock

@pytest.fixture
def mock_langchain_qdrant():
    """Fixture mockant l'intégration LangChain Qdrant."""
    with patch('app.services.vector_store.Qdrant') as mock:
        yield mock

@pytest.fixture
def mock_vector_store_embeddings():
    """Fixture mockant get_embedding_function pour vector_store."""
    with patch('app.services.vector_store.get_embedding_function') as mock:
        mock.return_value.embed_query.return_value = [0.1] * 1536
        yield mock
