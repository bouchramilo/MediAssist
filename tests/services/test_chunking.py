import pytest
from unittest.mock import patch, MagicMock
from app.services.chunking import split_documents

MODULE_PATH = "app.services.chunking"



@patch(f"{MODULE_PATH}.get_embeddings")
@patch(f"{MODULE_PATH}.SemanticChunker")
def test_split_documents_success(mock_splitter_class, mock_get_emb, mock_docs):
    
    
    mock_splitter_instance = mock_splitter_class.return_value
    
    chunk1 = MagicMock(page_content="Partie 1", metadata={"source": "pdf1"})
    chunk2 = MagicMock(page_content="Partie 2", metadata={"source": "pdf1"})
    mock_splitter_instance.split_documents.return_value = [chunk1, chunk2]

    result = split_documents(mock_docs)

    assert len(result) == 2
    assert result[0].metadata["chunk_id"] == 0
    assert result[1].metadata["chunk_id"] == 1
    assert result[0].metadata["chunk_type"] == "semantic"
    assert "chunk_length" in result[0].metadata
    
    mock_get_emb.assert_called_once()
    mock_splitter_class.assert_called_once()

def test_split_documents_empty_input():
    """Vérifie que la fonction gère une liste vide sans planter."""
    result = split_documents([])
    assert result == []

@patch(f"{MODULE_PATH}.get_embeddings")
def test_split_documents_error(mock_get_emb, mock_docs):
    """Vérifie qu'une RuntimeError est levée en cas d'échec interne."""
    
    mock_get_emb.side_effect = Exception("Modèle introuvable")

    with pytest.raises(RuntimeError) as excinfo:
        split_documents(mock_docs)
    
    assert "Chunking failed" in str(excinfo.value)