# app/services/rag_pipeline.py - Déjà bon avec vos modifications
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from app.services.prompt import get_prompt
from app.services.llm import create_llm
from app.services.vector_store import store_embeddings
from app.services.chunking import split_documents
from app.services.pdf_loader import load_pdf
from app.services.utils import format_docs
from app.utils.logger import AppLogger
from app.services.retriever import create_retriever

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

def initialize_rag_system(force_recreate_db=False, retriever_top_k: int = 5, retriever_alpha: float = 0.7):
    """
    Initialise le système RAG avec possibilité d'utiliser un retriever hybride.
    """
    logger.info("INITIALIZING RAG SYSTEM")

    if force_recreate_db:
        logger.info("Loading documents...")
        documents = load_pdf()
        print(f"🚩==========> nb_documents_loaded : ", len(documents))
        
        logger.info(f"Splitting {len(documents)} documents...")
        chunks = split_documents(documents)
        print(f"🚩==========> nb_chunks_created : ", len(chunks))
        
        logger.info("Storing embeddings in Qdrant...")
        store_embeddings(chunks)
    else:
        logger.info("Skipping document loading (force_recreate_db=False). Assuming VectorDB is populated.")

    # Utilisation de votre retriever hybride
    logger.info(f"Creating HybridRetriever with top_k={retriever_top_k}, alpha={retriever_alpha}")
    retriever = create_retriever(top_k=retriever_top_k, alpha=retriever_alpha)
    
    llm = create_llm()
    
    rag_chain = create_rag_chain(retriever, llm)

    return rag_chain