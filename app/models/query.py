# Table query
# Historique des questions/réponses
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.config.database import Base

class Query(Base):
    __tablename__ = "query"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, index=True)
    query = Column(String, nullable=False)
    response = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    user_id = Column(Integer, ForeignKey("users.id"))

    user = relationship("User", back_populates="queries")
