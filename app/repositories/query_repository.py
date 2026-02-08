# Sauvegarde des questions/réponses
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from app.models.query import Query
from typing import List, Dict, Any

def create_query_log(db: Session, query: str, response: str, user_id: int) -> Query:
    db_query = Query(query=query, response=response, user_id=user_id)
    db.add(db_query)
    db.commit()
    db.refresh(db_query)
    return db_query

def get_user_history(db: Session, user_id: int) -> List[Query]:
    return db.query(Query).filter(Query.user_id == user_id).order_by(desc(Query.created_at)).all()

def get_all_history(db: Session) -> List[Query]:
    return db.query(Query).order_by(desc(Query.created_at)).all()

def get_user_stats(db: Session, user_id: int) -> Dict[str, Any]:
    total_queries = db.query(Query).filter(Query.user_id == user_id).count()
    last_query = db.query(Query).filter(Query.user_id == user_id).order_by(desc(Query.created_at)).first()
    return {
        "total_queries": total_queries,
        "last_active": last_query.created_at if last_query else None
    }

def get_global_stats(db: Session) -> Dict[str, Any]:
    total_queries = db.query(Query).count()
    total_users_active = db.query(Query.user_id).distinct().count()
    
    frequent_questions = db.query(
        Query.query, func.count(Query.query).label("count")
    ).group_by(Query.query).order_by(desc("count")).limit(10).all()
    
    return {
        "total_queries": total_queries,
        "active_users_count": total_users_active,
        "frequent_questions": [{"question": q, "count": c} for q, c in frequent_questions]
    }
