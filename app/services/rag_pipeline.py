from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from app.services.prompt import get_prompt
from app.services.llm import create_llm
from app.services.vector_store import store_embeddings, search_semantic
from app.services.vector_store import get_vector_store
from app.services.chunking import split_documents
from app.services.pdf_loader import load_pdf
from app.services.utils import format_docs
from app.utils.logger import AppLogger

logger = AppLogger.get_logger(__name__)

def create_rag_chain(retriever, llm):
    prompt = get_prompt()
    
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def initialize_rag_system(force_recreate_db=False):
    logger.info("INITIALIZING RAG SYSTEM")
    
    if force_recreate_db:
        logger.info("Loading documents...")
        documents = load_pdf()
        
        logger.info(f"Splitting {len(documents)} documents...")
        chunks = split_documents(documents)
        
        logger.info("Storing embeddings in Qdrant...")
        store_embeddings(chunks)
    else:
        logger.info("Skipping document loading (force_recreate_db=False). Assuming VectorDB is populated.")
    
    vectorstore = get_vector_store()
    retriever = vectorstore.as_retriever()
    
    llm = create_llm()
    
    rag_chain = create_rag_chain(retriever, llm)
    
    logger.info("RAG SYSTEM READY!")
    
    return rag_chain
