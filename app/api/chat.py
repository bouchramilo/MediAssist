from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from app.config.database import get_db
from app.api.deps import get_current_active_user
from app.repositories.query_repository import create_query_log, get_user_history, get_user_stats
from app.models.user import User
from app.services.chat import ask_question as service_ask_question
from pydantic import BaseModel
from app.schemas.query import Query as QuerySchema

router = APIRouter(prefix="/chat", tags=["Chat"])

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]

@router.post("/", response_model=ChatResponse)
async def ask_question(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    # Call service
    response_data = await service_ask_question(request.question)
    
    # Log to DB
    create_query_log(
        db=db,
        query=request.question,
        response=response_data["answer"],
        user_id=current_user.id
    )
    
    return response_data

@router.get("/history", response_model=List[QuerySchema])
def get_my_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    return get_user_history(db, current_user.id)

@router.get("/stats", response_model=Dict[str, Any])
def get_my_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    return get_user_stats(db, current_user.id)
