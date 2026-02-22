# app/services/chat.py

from app.services.rag_pipeline import initialize_rag_system
from app.utils.logger import AppLogger
from app.mlops.evaluation import evaluate_rag
from app.mlops import tracking
import mlflow
from datetime import datetime
import json
import time

from app.metrics import (
    RAG_REQUEST_TOTAL, 
    RAG_PROCESSING_TIME, 
    RAG_METRIC_FAITHFULNESS, 
    RAG_METRIC_ANSWER_RELEVANCE,
    RAG_DOCS_RETRIEVED
)

logger = AppLogger.get_logger(__name__)

_qa_chain = None

def get_qa_chain(force_recreate_db=False, use_hybrid=True, top_k=5, alpha=0.7):
    """
    Initialise ou récupère la chaîne RAG.
    """
    global _qa_chain
    if _qa_chain is None or force_recreate_db:
        logger.info("Initializing RAG chain...")
        if use_hybrid:
            _qa_chain = initialize_rag_system(
                force_recreate_db=force_recreate_db,
                retriever_top_k=top_k,
                retriever_alpha=alpha
            )
        else:
            _qa_chain = initialize_rag_system(force_recreate_db=force_recreate_db)
    return _qa_chain

async def ask_question(question: str, top_k: int = 5, alpha: float = 0.7):
    """
    Pose une question au système RAG et retourne la réponse avec les sources.
    """
    start_time = time.time()
    try:
        RAG_REQUEST_TOTAL.labels(status="success").inc()
        logger.info("📈 Prometheus Metric: rag_request_total(status='success') incremented")
        chain = get_qa_chain(
            force_recreate_db=False,
            use_hybrid=True,
            top_k=top_k,
            alpha=alpha
        )
        
        # Exécution de la chaîne RAG
        res = chain.invoke(question)
        
        answer = res["answer"]
        source_docs = res["context"]
        num_docs = len(source_docs)
        RAG_DOCS_RETRIEVED.observe(num_docs)
        logger.info(f"🔍 Prometheus Metric: rag_docs_retrieved_count observed value: {num_docs}")
        
        # Extraction des sources uniques
        sources = list(set([
            doc.metadata.get("source", "unknown") 
            for doc in source_docs
        ]))
        
        # MLOps: Évaluation et Logging
        mlflow_logger = None
        query_run = None
        
        try:
            # Création d'un run MLflow dédié à cette requête
            mlflow_logger, query_run = tracking.create_query_run(
                run_name_prefix=f"query_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            )
            run_id = query_run.info.run_id
            
            logger.info(f"MLOps Run started. Run ID: {run_id}")
            
            # Extraction des textes des chunks pour l'évaluation
            chunk_texts = [doc.page_content for doc in source_docs]
            
            # Évaluation avec DeepEval
            metrics = evaluate_rag(
                query=question,
                response=answer,
                context=chunk_texts
            )
            
            # Log des métriques
            mlflow_logger.log_metrics(metrics)
            
            # Prometheus Metrics Update
            if "Faithfulness" in metrics:
                val = metrics["Faithfulness"]
                RAG_METRIC_FAITHFULNESS.labels(model="rag_v1").set(val)
                logger.info(f"⚖️ Prometheus Metric: rag_metric_faithfulness set to {val}")
            if "Answer Relevance" in metrics:
                val = metrics["Answer Relevance"]
                RAG_METRIC_ANSWER_RELEVANCE.labels(model="rag_v1").set(val)
                logger.info(f"🎯 Prometheus Metric: rag_metric_answer_relevance set to {val}")
            
            # Log des paramètres du retriever
            mlflow_logger.log_rag_config({
                "retriever_type": "hybrid",
                "retriever_top_k": top_k,
                "retriever_alpha": alpha,
                "timestamp": datetime.now().isoformat()
            })
            
            # === ARTEFACT 1: Conversation complète ===
            conversation_data = {
                "query": question,
                "response": answer,
                "timestamp": datetime.now().isoformat(),
                "retriever_config": {
                    "top_k": top_k,
                    "alpha": alpha
                },
                "metrics": metrics
            }
            
            mlflow_logger.log_text(
                json.dumps(conversation_data, indent=2, ensure_ascii=False),
                "conversation.json"
            )
            
            # Format texte lisible
            conversation_text = f"""================================================================
                QUESTION: {question}
                ================================================================
                RÉPONSE: {answer}
                ================================================================
                TIMESTAMP: {datetime.now().isoformat()}
                CONFIG: top_k={top_k}, alpha={alpha}
                MÉTRIQUES: {json.dumps(metrics, indent=2)}
                ================================================================
                """
            mlflow_logger.log_text(conversation_text, "conversation.txt")
            
            # === ARTEFACT 2: Contexte complet (tous les chunks) ===
            context_data = {
                "query": question,
                "retrieved_documents": []
            }
            
            for i, doc in enumerate(source_docs, 1):
                doc_info = {
                    "rank": i,
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "N/A"),
                    "chapter": doc.metadata.get("chapter", "N/A"),
                    "section": doc.metadata.get("section", "N/A"),
                    "content": doc.page_content,
                    "content_length": len(doc.page_content)
                }
                context_data["retrieved_documents"].append(doc_info)
                
            
            # Sauvegarde en JSON pour exploitation future
            mlflow_logger.log_text(
                json.dumps(context_data, indent=2),
                "retrieved_context.json"
            )
            logger.info(f"📋📋 retrieved_context.json : {context_data}")
            
            # Version texte résumée pour consultation rapide
            context_summary = f"CONTEXT POUR LA QUESTION: {question}\n"
            context_summary += f"Nombre de documents récupérés: {len(source_docs)}\n"
            context_summary += "=" * 50 + "\n\n"
            
            for i, doc in enumerate(source_docs, 1):
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "N/A")
                chapter = doc.metadata.get("chapter", "N/A")
                section = doc.metadata.get("section", "N/A")
                content_preview = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                
                context_summary += f"[Document {i}]\n"
                context_summary += f"Source: {source} (Page {page})\n"
                if chapter != "N/A":
                    context_summary += f"Chapitre: {chapter}\n"
                if section != "N/A":
                    context_summary += f"Section: {section}\n"
                context_summary += f"Contenu:\n{content_preview}\n"
                context_summary += "-" * 30 + "\n\n"
            
            mlflow_logger.log_text(context_summary, "context_summary.txt")
            logger.info(f"📋📋 context_summary.txt : {context_summary}")
            
            # === ARTEFACT 3: Statistiques de la requête ===
            stats = {
                "query_length": len(question),
                "response_length": len(answer),
                "num_docs_retrieved": len(source_docs),
                "unique_sources": len(sources),
                "avg_doc_length": sum(len(doc.page_content) for doc in source_docs) / len(source_docs) if source_docs else 0,
                "retriever_top_k": top_k,
                "retriever_alpha": alpha
            }
            
            mlflow_logger.log_text(
                json.dumps(stats, indent=2),
                "query_stats.json"
            )
            logger.info(f"📋📋 query_stats.json : {stats}")
            
            
            
            logger.info(f"MLOps artifacts logged successfully for run {run_id}")
                
        except Exception as e:
            logger.warning(f"MLOps Evaluation failed: {e}")
            import traceback
            logger.warning(traceback.format_exc())
        finally:
            if mlflow_logger:
                mlflow_logger.end_run()
                logger.info("MLOps Run ended.")

        return {
            "answer": answer,
            "sources": sources,
            "num_chunks": len(source_docs)
        }
        
    except Exception as e:
        RAG_REQUEST_TOTAL.labels(status="error").inc()
        logger.info("❌ Prometheus Metric: rag_request_total(status='error') incremented")
        logger.error(f"Error in ask_question: {e}")
        return {
            "answer": f"Une erreur est survenue lors du traitement de votre demande : {e}",
            "sources": [],
            "num_chunks": 0
        }
    finally:
        total_time = time.time() - start_time
        RAG_PROCESSING_TIME.observe(total_time)