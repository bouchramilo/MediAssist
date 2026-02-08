from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from app.config.database import get_db
from app.api.deps import get_current_admin_user
from app.repositories.query_repository import get_global_stats, get_all_history, get_user_history
from app.models.user import User
from app.schemas.query import Query as QuerySchema

router = APIRouter(prefix="/admin", tags=["Admin"])

@router.get("/stats", response_model=Dict[str, Any])
def get_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    return get_global_stats(db)

@router.get("/history", response_model=List[QuerySchema])
def get_global_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    return get_all_history(db)

@router.get("/users/{user_id}/history", response_model=List[QuerySchema])
def get_specific_user_history(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    return get_user_history(db, user_id)
