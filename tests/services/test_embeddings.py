import pytest
from unittest.mock import patch, Mock

from app.services.embeddings import get_embedding_function
from app.config import settings


def test_get_embedding_function():
    """Test la création de la fonction d'embedding"""
    
    with patch('app.services.embeddings.OllamaEmbeddings') as mock_embeddings:
        mock_instance = Mock()
        mock_embeddings.return_value = mock_instance
        
        # Appel de la fonction
        result = get_embedding_function()
        
        # Vérifications
        mock_embeddings.assert_called_once_with(
            model="nomic-embed-text",
            base_url=settings.OLLAMA_BASE_URL
        )
        assert result == mock_instance
        assert result is get_embedding_function()


def test_get_embedding_function_custom_model():
    """Test avec un modèle personnalisé"""
    
    with patch('app.services.embeddings.OllamaEmbeddings') as mock_embeddings:
        mock_instance = Mock()
        mock_embeddings.return_value = mock_instance
        
        result = get_embedding_function("custom-model")
        
        mock_embeddings.assert_called_once_with(
            model="custom-model",
            base_url=settings.OLLAMA_BASE_URL
        )
        assert result == mock_instance


def test_get_embedding_function_error():
    """Test la gestion d'erreur"""
    get_embedding_function.cache_clear()
    
    with patch('app.services.embeddings.OllamaEmbeddings') as mock_embeddings:
        mock_embeddings.side_effect = Exception("Connection failed")
        
        with pytest.raises(Exception, match="Connection failed"):
            get_embedding_function()


def test_get_embedding_function_cache():
    """Test spécifique du cache LRU"""
    
    with patch('app.services.embeddings.OllamaEmbeddings') as mock_embeddings:
        mock_instance1 = Mock(name="instance1")
        mock_instance2 = Mock(name="instance2")
        
        mock_embeddings.return_value = mock_instance1
        result1 = get_embedding_function("model1")
        
        mock_embeddings.return_value = mock_instance2
        result2 = get_embedding_function("model2")
        
        result3 = get_embedding_function("model1")
        
        # Vérifications
        assert result1 == mock_instance1
        assert result2 == mock_instance2
        assert result3 == mock_instance1
        assert result1 is result3 
        assert result1 is not result2
    
        assert mock_embeddings.call_count == 2