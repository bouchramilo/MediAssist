# CRUD utilisateurs
from app.schemas.user import UserCreate, UserInDBBase
from app.models.user import User
from sqlalchemy.orm import Session
from app.security.password import hash_password 
from typing import List


def check_user_existe(db: Session, email: str, username: str):
    return (
        db.query(User)
        .filter((User.email == email) | (User.username == username))
        .first()
    )
    
    
    
def create_user(db:Session, user:UserCreate):
    db_user = User(
        email=user.email,
        username=user.username,
        password_hash=hash_password(user.password),
        role=user.role,
        is_active=True
    )    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user




def get_user_by_username(db: Session, username: str):
    return db.query(User).filter((User.username == username) | (User.email == username)).first()

def get_all_users(db: Session) -> List[UserInDBBase]:
    return db.query(User).all()
