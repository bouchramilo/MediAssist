import pytest
from unittest.mock import patch, MagicMock
from app.services.chat import ask_question, get_qa_chain

@patch("app.services.chat.initialize_rag_system")
@pytest.mark.asyncio
async def test_ask_question_first_call(mock_init):
    # Setup - First call should initialize the chain
    mock_chain = MagicMock()
    mock_init.return_value = mock_chain
    
    doc = MagicMock()
    doc.metadata = {"source": "doc1.pdf"}
    
    mock_chain.invoke.return_value = {
        "answer": "Test answer",
        "context": [doc]
    }
    
    # Execute - Need to reset singleton first
    import app.services.chat
    app.services.chat._qa_chain = None
    
    result = await ask_question("Hello")
    
    # Assert
    mock_init.assert_called_once()
    mock_chain.invoke.assert_called_once_with("Hello")
    assert result["answer"] == "Test answer"
    assert result["sources"] == ["doc1.pdf"]

@patch("app.services.chat.initialize_rag_system")
@pytest.mark.asyncio
async def test_ask_question_error(mock_init):
    # Setup
    mock_chain = MagicMock()
    mock_init.return_value = mock_chain
    mock_chain.invoke.side_effect = Exception("RAG Error")
    
    # Reset singleton
    import app.services.chat
    app.services.chat._qa_chain = None
    
    # Execute
    result = await ask_question("Hello")
    
    # Assert
    assert "Une erreur est survenue" in result["answer"]
    assert result["sources"] == []
