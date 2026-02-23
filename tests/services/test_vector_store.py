from unittest.mock import Mock, patch, MagicMock
import pytest
from app.services.vector_store import (
    create_qdrant_collection,
    store_embeddings,
    get_vector_store,
    search_semantic,
    search_keyword,
    search_hybrid
)
from app.config import settings
from qdrant_client.http import models

def test_create_qdrant_collection_exists(mock_qdrant_client, mock_vector_store_embeddings):
    client_instance = mock_qdrant_client.return_value
    mock_collection = Mock()
    mock_collection.name = settings.QDRANT_COLLECTION_NAME
    client_instance.get_collections.return_value.collections = [mock_collection]
    
    # Execute
    result = create_qdrant_collection()
    
    assert result is True
    client_instance.create_collection.assert_not_called()

def test_create_qdrant_collection_new(mock_qdrant_client, mock_vector_store_embeddings):
    client_instance = mock_qdrant_client.return_value
    client_instance.get_collections.return_value.collections = []
    
    # Execute
    result = create_qdrant_collection()
    
    assert result is True
    client_instance.create_collection.assert_called_once()
    mock_vector_store_embeddings.return_value.embed_query.assert_called_with("test")

def test_store_embeddings(mock_qdrant_client, mock_vector_store_embeddings, mock_langchain_qdrant):
    
    client_instance = mock_qdrant_client.return_value

    mock_collection = Mock()
    mock_collection.name = settings.QDRANT_COLLECTION_NAME
    client_instance.get_collections.return_value.collections = [mock_collection]
    
    chunks = [Mock(page_content="test", metadata={})]
    
    # Execute
    result = store_embeddings(chunks)
    
    assert result is True
    mock_langchain_qdrant.from_documents.assert_called_once()

def test_get_vector_store(mock_qdrant_client, mock_vector_store_embeddings, mock_langchain_qdrant):
    # Execute
    result = get_vector_store()
    
    mock_langchain_qdrant.assert_called_once()
    assert result == mock_langchain_qdrant.return_value

def test_search_semantic(mock_qdrant_client, mock_vector_store_embeddings, mock_langchain_qdrant):
    
    mock_store = mock_langchain_qdrant.return_value
    mock_doc = Mock(page_content="result")
    mock_store.similarity_search_with_score.return_value = [(mock_doc, 0.9)]
    
    # Execute
    results = search_semantic("query")
    
    assert len(results) == 1
    assert results[0][0] == mock_doc
    mock_store.similarity_search_with_score.assert_called_with(
        query="query",
        k=10,
        filter=None
    )

def test_search_keyword(mock_qdrant_client):
    
    client_instance = mock_qdrant_client.return_value
    mock_point = Mock()
    mock_point.payload = {'page_content': 'this is a test keyword result', 'metadata': {}}
    client_instance.scroll.return_value = ([mock_point], None)
    
    # Execute
    results = search_keyword("test keyword")
    
    assert len(results) == 1
    assert results[0][0].page_content == 'this is a test keyword result'
    client_instance.scroll.assert_called_once()

def test_search_hybrid(mock_qdrant_client, mock_vector_store_embeddings, mock_langchain_qdrant):
    
    mock_store = mock_langchain_qdrant.return_value
    mock_doc_sem = {'page_content': 'semantic'} 

    mock_doc_obj = Mock()
    mock_doc_obj.page_content = "semantic_doc"
    mock_store.similarity_search_with_score.return_value = [(mock_doc_obj, 0.8)]
    
    # Mock keyword search
    client_instance = mock_qdrant_client.return_value
    mock_point = Mock()
    mock_point.payload = {'page_content': 'keyword_doc', 'metadata': {}}
    client_instance.scroll.return_value = ([mock_point], None)
    
    # Execute
    results = search_hybrid("test query", top_k=2)
    
    assert len(results) <= 2
    assert len(results) > 0
