import pytest
from sqlalchemy import text
from app.config.database import engine, SessionLocal, get_db

def test_database_connection():
    """Test de connexion simple à la base de données."""
    try:
        # Tenter une connexion
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            assert result.scalar() == 1
            print("\nConnexion Database: OK")
    except Exception as e:
        pytest.fail(f"Échec de la connexion à la base de données: {str(e)}")

def test_session_creation():
    """Test de création de session via SessionLocal."""
    db = SessionLocal()
    assert db is not None
    db.close()
    print("Création Session (SessionLocal): OK")

def test_get_db_dependency():
    """Test du générateur get_db used for dependency injection."""
    gen = get_db()
    db = next(gen)
    assert db is not None

    try:
        pass 
    finally:

        try:
            next(gen)
        except StopIteration:
            pass
    print("Dependency get_db: OK")
