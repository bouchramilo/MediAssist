import pytest
from unittest.mock import patch, MagicMock
from app.services.rag_pipeline import initialize_rag_system, create_rag_chain

@patch("app.services.rag_pipeline.load_pdf")
@patch("app.services.rag_pipeline.split_documents")
@patch("app.services.rag_pipeline.store_embeddings")
@patch("app.services.rag_pipeline.get_vector_store")
@patch("app.services.rag_pipeline.create_llm")
def test_initialize_rag_system_force_recreate(
    mock_create_llm, 
    mock_get_vector_store, 
    mock_store_embeddings, 
    mock_split_documents, 
    mock_load_pdf
):
    # Setup Mocks
    mock_load_pdf.return_value = ["doc1"]
    mock_split_documents.return_value = ["chunk1"]
    mock_vector_store = MagicMock()
    mock_get_vector_store.return_value = mock_vector_store
    mock_llm = MagicMock()
    mock_create_llm.return_value = mock_llm
    
    # Execute
    chain = initialize_rag_system(force_recreate_db=True)
    
    # Assert
    mock_load_pdf.assert_called_once()
    mock_split_documents.assert_called_once()
    mock_store_embeddings.assert_called_once()
    mock_get_vector_store.assert_called_once()
    mock_create_llm.assert_called_once()
    assert chain is not None

@patch("app.services.rag_pipeline.load_pdf")
@patch("app.services.rag_pipeline.split_documents")
@patch("app.services.rag_pipeline.store_embeddings")
@patch("app.services.rag_pipeline.get_vector_store")
@patch("app.services.rag_pipeline.create_llm")
def test_initialize_rag_system_no_recreate(
    mock_create_llm, 
    mock_get_vector_store, 
    mock_store_embeddings, 
    mock_split_documents, 
    mock_load_pdf
):
    # Setup Mocks
    mock_vector_store = MagicMock()
    mock_get_vector_store.return_value = mock_vector_store
    mock_llm = MagicMock()
    mock_create_llm.return_value = mock_llm
    
    # Execute
    chain = initialize_rag_system(force_recreate_db=False)
    
    # Assert
    mock_load_pdf.assert_not_called()
    mock_split_documents.assert_not_called()
    mock_store_embeddings.assert_not_called()
    mock_get_vector_store.assert_called_once()
    assert chain is not None
