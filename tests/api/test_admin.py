from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from app.main import app
from app.models.user import User
from app.config.database import get_db
from app.api.deps import get_current_admin_user

client = TestClient(app)

def test_get_stats():
    # Mock database and user
    mock_db = MagicMock()
    mock_user = User(id=1, email="admin@test.com", username="admin", role="ADMIN", is_active=True)
    
    # Mock repository function
    mock_stats = {"total_users": 10, "total_queries": 50}
    
    app.dependency_overrides[get_db] = lambda: mock_db
    app.dependency_overrides[get_current_admin_user] = lambda: mock_user
    
    try:
        with patch("app.api.admin.get_global_stats", return_value=mock_stats):
            response = client.get("/api/v1/admin/stats")
            
            assert response.status_code == 200
            assert response.json() == mock_stats
    finally:
        app.dependency_overrides = {}

def test_get_global_history():
    mock_db = MagicMock()
    mock_user = User(id=1, email="admin@test.com", username="admin", role="ADMIN", is_active=True)
    mock_history = [{"id": 1, "query": "q1", "response": "a1", "user_id": 1, "timestamp": "2023-01-01"}]
    
    app.dependency_overrides[get_db] = lambda: mock_db
    app.dependency_overrides[get_current_admin_user] = lambda: mock_user

    try:
        with patch("app.api.admin.get_all_history", return_value=mock_history):
            response = client.get("/api/v1/admin/history")
            
            assert response.status_code == 200
            assert len(response.json()) == 1
            assert response.json()[0]["query"] == "q1"
    finally:
        app.dependency_overrides = {}

def test_get_specific_user_history():
    mock_db = MagicMock()
    mock_user = User(id=1, email="admin@test.com", username="admin", role="ADMIN", is_active=True)
    mock_history = [{"id": 1, "query": "q1", "response": "a1", "user_id": 2, "timestamp": "2023-01-01"}]
    
    app.dependency_overrides[get_db] = lambda: mock_db
    app.dependency_overrides[get_current_admin_user] = lambda: mock_user

    try:
        with patch("app.api.admin.get_user_history", return_value=mock_history):
            response = client.get("/api/v1/admin/users/2/history")
            
            assert response.status_code == 200
            assert len(response.json()) == 1
            assert response.json()[0]["user_id"] == 2
    finally:
        app.dependency_overrides = {}
