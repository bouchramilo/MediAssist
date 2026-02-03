# Découpe le texte en chunks
# Ajoute métadonnées (page, source)
from app.utils.logger import AppLogger
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings


logger = AppLogger.get_logger(__name__)

# ! ==================================================================================
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )



# ! ==================================================================================
def split_documents(documents):
    if not documents:
        logger.warning("No documents provided for chunking.")
        return []

    logger.info(f"Splitting {len(documents)} documents into chunks (semantic chunking)...")
    
    try:
        embeddings = get_embeddings()
        text_splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95
        )
        
        chunks = text_splitter.split_documents(documents)
        
        # q;2liorer le métadonnées
        for idx, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_id": idx,
                "chunk_length": len(chunk.page_content),
                "chunk_type": "semantic",
            })
            
        logger.info(f"Chunking completed: {len(chunks)} chunks created.")
        return chunks
        
    except Exception as e:
        logger.exception("❌ Error during document chunking")
        raise RuntimeError("Chunking failed") from e
    
    
# ! ==================================================================================