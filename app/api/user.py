from fastapi import APIRouter, Depends, status, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.schemas.user import UserCreate, User, UserInDBBase
from app.repositories.user_repository import (
    check_user_existe,
    create_user,
    get_all_users
)
from app.config.database import get_db
from fastapi.security import OAuth2PasswordRequestForm
from app.security.jwt import create_access_token
from app.security.password import verify_password
from app.repositories.user_repository import get_user_by_username
from app.schemas.auth import Token
from app.api.deps import get_current_admin_user

router = APIRouter(prefix="/users", tags=["Users"])


@router.post("/register", response_model=User, status_code=status.HTTP_201_CREATED)
def register_user(user: UserCreate, db: Session = Depends(get_db)):

    db_user = check_user_existe(
        db=db,
        email=user.email,
        username=user.username,
    )

    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email or username already exists"
        )

    return create_user(db=db, user=user)


@router.post("/login", response_model=Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = get_user_by_username(db, username=form_data.username)
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(subject=user.username)
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/all", response_model=List[UserInDBBase])
def get_users(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    return get_all_users(db)
