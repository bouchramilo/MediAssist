from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient
from app.main import app
from app.api.deps import get_current_active_user, get_db
from app.models.user import User

client = TestClient(app)

def test_ask_question():
    # Mock database
    mock_db = MagicMock()
    
    # Mock current user
    mock_user = User(id=1, email="user@test.com", username="testuser", role="USER", is_active=True)
    
    # Mock service response
    mock_service_response = {"answer": "Test answer", "sources": ["source1"]}
    
    app.dependency_overrides[get_db] = lambda: mock_db
    app.dependency_overrides[get_current_active_user] = lambda: mock_user

    try:
        with patch("app.api.chat.service_ask_question", new_callable=AsyncMock) as mock_service, \
             patch("app.api.chat.create_query_log") as mock_log:
            
            mock_service.return_value = mock_service_response
            
            response = client.post("/api/v1/chat/", json={"question": "Test question"})
            
            assert response.status_code == 200
            assert response.json() == mock_service_response
            mock_service.assert_called_once_with("Test question")
            mock_log.assert_called_once()
    finally:
        app.dependency_overrides = {}

def test_get_my_history():
    mock_db = MagicMock()
    mock_user = User(id=1, email="user@test.com", username="testuser", role="USER", is_active=True)
    mock_history = [{"id": 1, "query": "q1", "response": "a1", "user_id": 1, "timestamp": "2023-01-01"}]
    
    app.dependency_overrides[get_db] = lambda: mock_db
    app.dependency_overrides[get_current_active_user] = lambda: mock_user

    try:
        with patch("app.api.chat.get_user_history", return_value=mock_history):
            response = client.get("/api/v1/chat/history")
            
            assert response.status_code == 200
            # Check if the list contains the expected item
            assert len(response.json()) == 1
            assert response.json()[0]["query"] == "q1"
    finally:
        app.dependency_overrides = {}

def test_get_my_stats():
    mock_db = MagicMock()
    mock_user = User(id=1, email="user@test.com", username="testuser", role="USER", is_active=True)
    mock_stats = {"total_queries": 10}
    
    app.dependency_overrides[get_db] = lambda: mock_db
    app.dependency_overrides[get_current_active_user] = lambda: mock_user

    try:
        with patch("app.api.chat.get_user_stats", return_value=mock_stats):
            response = client.get("/api/v1/chat/stats")
            
            assert response.status_code == 200
            assert response.json() == mock_stats
    finally:
        app.dependency_overrides = {}
