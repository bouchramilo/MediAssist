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


@router.get("/all", response_model=List[UserInDBBase])
def get_users(db: Session = Depends(get_db)):
    return get_all_users(db)
