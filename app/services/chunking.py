

import re
from typing import List, Dict, Optional
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document
from app.utils.logger import AppLogger

logger = AppLogger.get_logger(__name__)

def estimate_tokens(text: str) -> int:
    return len(text.split())

def split_by_paragraph(text: str, max_tokens: int = 500, overlap: int = 80) -> List[str]:
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for p in paragraphs:
        if estimate_tokens(current + p) <= max_tokens:
            current += "\n\n" + p if current else p
        else:
            if current:
                chunks.append(current.strip())
            
            words = current.split()
            tail = " ".join(words[-overlap:]) if overlap < len(words) else current
            current = tail + "\n\n" + p

    if current.strip():
        chunks.append(current.strip())

    return chunks

def chunk_markdown_document(
    text: str,
    source: str,
    page: int = 1,
    max_tokens: int = 500
) -> List[Dict]:
    """
    Split a markdown text based on headers (## or ###) and then by paragraphs if needed.
    """
    sections = re.split(r"(?=^## |\n## |\n### )", text, flags=re.MULTILINE)

    chunks = []
    current_chapter = None
    current_section = None

    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue

        chap_match = re.search(r"^#\s+(.+)", sec, re.MULTILINE)
        if chap_match:
            current_chapter = chap_match.group(1).strip()
        
        sec_match = re.search(r"^(##+)\s+(.+)", sec, re.MULTILINE)
        if sec_match:
            current_section = sec_match.group(2).strip()

        if estimate_tokens(sec) > max_tokens:
            sub_chunks = split_by_paragraph(sec, max_tokens=max_tokens)
        else:
            sub_chunks = [sec]

        for chunk_text in sub_chunks:
            chunks.append({
                "content": chunk_text,
                "metadata": {
                    "source": source,
                    "page": page,
                    "chapter": current_chapter,
                    "section": current_section,
                    "chunk_type": "hierarchical"
                }
            })

    return chunks

def split_documents(documents: List[Document]) -> List[Document]:
    if not documents:
        logger.warning("No documents provided for chunking.")
        return []

    logger.info(f"Splitting {len(documents)} documents into chunks (hierarchical)...")
    
    all_chunks = []
    
    try:
        for doc in documents:
            
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", 1)
            
            chunk_dicts = chunk_markdown_document(
                text=doc.page_content, 
                source=source, 
                page=page, 
                max_tokens=500
            )
            
            for chunk_data in chunk_dicts:
                combined_metadata = doc.metadata.copy()
                combined_metadata.update(chunk_data["metadata"])
                
                new_doc = Document(
                    page_content=chunk_data["content"],
                    metadata=combined_metadata
                )
                all_chunks.append(new_doc)
            
        logger.info(f"Chunking completed: {len(all_chunks)} chunks created.")
        return all_chunks
        
    except Exception as e:
        logger.exception("❌ Error during document chunking")
        raise RuntimeError("Chunking failed") from e
