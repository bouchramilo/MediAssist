from unittest.mock import Mock, patch
from app.services.retriever import HybridRetriever, create_retriever
try:
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
except ImportError:
    from langchain.callbacks.manager import CallbackManagerForRetrieverRun

@patch("app.services.retriever.search_hybrid")
def test_hybrid_retriever_get_relevant_documents(mock_search_hybrid):
    mock_doc1 = Mock()
    mock_doc1.page_content = "doc1"
    mock_doc2 = Mock()
    mock_doc2.page_content = "doc2"
    
    mock_search_hybrid.return_value = [mock_doc1, mock_doc2]
    
    retriever = HybridRetriever(top_k=5, alpha=0.7)
    
    mock_run_manager = Mock(spec=CallbackManagerForRetrieverRun)
    
    results = retriever._get_relevant_documents("test query", run_manager=mock_run_manager)
    
    assert len(results) == 2
    assert results[0] == mock_doc1
    assert results[1] == mock_doc2
    
    mock_search_hybrid.assert_called_once_with(
        query="test query",
        top_k=5,
        alpha=0.7
    )

@patch("app.services.retriever.search_hybrid")
def test_hybrid_retriever_error_handling(mock_search_hybrid):
    mock_search_hybrid.side_effect = Exception("Search error")
    
    retriever = HybridRetriever()
    mock_run_manager = Mock(spec=CallbackManagerForRetrieverRun)
    
    results = retriever._get_relevant_documents("test query", run_manager=mock_run_manager)
    
    assert results == []
    mock_search_hybrid.assert_called_once()

def test_create_retriever():
    retriever = create_retriever(top_k=10, alpha=0.5)
    
    assert isinstance(retriever, HybridRetriever)
    assert retriever.top_k == 10
    assert retriever.alpha == 0.5
