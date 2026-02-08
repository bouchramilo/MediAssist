# Schémas pour requête utilisateur et réponse RAG
from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime


class QueryBase(BaseModel):
    query: str = Field(..., min_length=1, description="Question de l'utilisateur")
    response: str = Field(..., min_length=1, description="Réponse du système RAG")
    created_at: Optional[datetime] = None
    user_id: int = Field(..., gt=0, description="ID de l'utilisateur")


class QueryCreate(BaseModel):
    query: str = Field(..., min_length=1, description="Question de l'utilisateur")
    response: str = Field(..., min_length=1, description="Réponse du système RAG")
    user_id: int = Field(..., gt=0, description="ID de l'utilisateur")


class QueryUpdate(BaseModel):
    query: Optional[str] = Field(None, min_length=1, description="Question de l'utilisateur")
    response: Optional[str] = Field(None, min_length=1, description="Réponse du système RAG")


class QueryInDBBase(QueryBase):
    id: int
    
    class Config:
        from_attributes = True


class Query(QueryInDBBase):
    pass