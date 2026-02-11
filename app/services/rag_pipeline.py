from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from app.services.prompt import get_prompt
from app.services.llm import create_llm
from app.services.vector_store import store_embeddings, search_semantic
from app.services.vector_store import get_vector_store
from app.services.chunking import split_documents
from app.services.pdf_loader import load_pdf
from app.services.utils import format_docs
from app.utils.logger import AppLogger
from app.mlops.tracking import log_rag_experiment
import mlflow

logger = AppLogger.get_logger(__name__)

def create_rag_chain(retriever, llm):
    prompt = get_prompt()
    
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    return rag_chain_with_source


def initialize_rag_system(force_recreate_db=False):
    logger.info("INITIALIZING RAG SYSTEM")

    # Initialize MLOps Tracking
    mlflow_logger, run = log_rag_experiment(run_name="rag_pipeline")
    
    mlflow_logger.log_rag_config({"force_recreate_db": force_recreate_db})

    if force_recreate_db:
        logger.info("Loading documents...")
        documents = load_pdf()
        mlflow_logger.log_metrics({"nb_documents_loaded": len(documents)})
        print(f"🚩==========> nb_documents_loaded : ", len(documents))
        
        logger.info(f"Splitting {len(documents)} documents...")
        chunks = split_documents(documents)
        mlflow_logger.log_metrics({"nb_chunks_created": len(chunks)})
        print(f"🚩==========> nb_chunks_created : ", len(chunks))
        
        logger.info("Storing embeddings in Qdrant...")
        store_embeddings(chunks)
    else:
        logger.info("Skipping document loading (force_recreate_db=False). Assuming VectorDB is populated.")

    vectorstore = get_vector_store()
    retriever = vectorstore.as_retriever()
    
    llm = create_llm()
    
    rag_chain = create_rag_chain(retriever, llm)

    

    return rag_chain
