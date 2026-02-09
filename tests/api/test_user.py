from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from app.main import app
from app.api.deps import get_db, get_current_admin_user
from app.models.user import User

client = TestClient(app)

def test_register_user_success():
    mock_db = MagicMock()
    user_data = {"email": "new@test.com", "username": "newuser", "password": "password123", "password_repeat": "password123"}
    
    app.dependency_overrides[get_db] = lambda: mock_db
    
    try:
        with patch("app.api.user.check_user_existe", return_value=None), \
             patch("app.api.user.create_user") as mock_create:
            
            user_data_model = user_data.copy()
            del user_data_model["password"]
            del user_data_model["password_repeat"]
            mock_create.return_value = User(id=1, **user_data_model, password_hash="hashed", role="USER", is_active=True, created_at="2023-01-01T00:00:00")
            
            response = client.post("/api/v1/users/register", json=user_data)
            
            assert response.status_code == 201
            assert response.json()["email"] == user_data["email"]
    finally:
        app.dependency_overrides = {}

def test_register_user_exists():
    mock_db = MagicMock()
    user_data = {"email": "existing@test.com", "username": "existing", "password": "password123", "password_repeat": "password123"}
    
    app.dependency_overrides[get_db] = lambda: mock_db
    
    try:
        with patch("app.api.user.check_user_existe") as mock_check:
            user_data_model = user_data.copy()
            del user_data_model["password"]
            del user_data_model["password_repeat"]
            mock_check.return_value = User(id=1, **user_data_model, password_hash="hashed", role="USER", is_active=True, created_at="2023-01-01T00:00:00")
            
            response = client.post("/api/v1/users/register", json=user_data)
            
            assert response.status_code == 400
            assert "already exists" in response.json()["detail"]
    finally:
        app.dependency_overrides = {}

def test_login_success():
    mock_db = MagicMock()
    login_data = {"username": "testuser", "password": "password123"}
    mock_user = User(id=1, email="user@test.com", username="testuser", password_hash="hashed_pw")
    
    app.dependency_overrides[get_db] = lambda: mock_db

    try:
        with patch("app.api.user.get_user_by_username", return_value=mock_user), \
             patch("app.api.user.verify_password", return_value=True), \
             patch("app.api.user.create_access_token", return_value="fake_token"):
            
            response = client.post("/api/v1/users/login", data=login_data)
            
            assert response.status_code == 200
            assert response.json()["access_token"] == "fake_token"
    finally:
        app.dependency_overrides = {}

def test_login_failure():
    mock_db = MagicMock()
    login_data = {"username": "testuser", "password": "wrongpassword"}
    mock_user = User(id=1, email="user@test.com", username="testuser", password_hash="hashed_pw")
    
    app.dependency_overrides[get_db] = lambda: mock_db

    try:
        with patch("app.api.user.get_user_by_username", return_value=mock_user), \
             patch("app.api.user.verify_password", return_value=False):
            
            response = client.post("/api/v1/users/login", data=login_data)
            
            assert response.status_code == 401
            assert "Incorrect username or password" in response.json()["detail"]
    finally:
        app.dependency_overrides = {}

def test_get_users_admin():
    mock_db = MagicMock()
    mock_admin = User(id=1, email="admin@test.com", username="admin", role="ADMIN", is_active=True)
    mock_users_list = [
        User(id=1, email="admin@test.com", username="admin", role="ADMIN", password_hash="hash", is_active=True, created_at="2023-01-01T00:00:00"),
        User(id=2, email="user@test.com", username="user", role="USER", password_hash="hash", is_active=True, created_at="2023-01-01T00:00:00")
    ]
    
    app.dependency_overrides[get_db] = lambda: mock_db
    app.dependency_overrides[get_current_admin_user] = lambda: mock_admin

    try:
        with patch("app.api.user.get_all_users", return_value=mock_users_list):
            response = client.get("/api/v1/users/all")
            
            assert response.status_code == 200
            assert len(response.json()) == 2
    finally:
        app.dependency_overrides = {}
