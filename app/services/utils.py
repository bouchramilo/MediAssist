from typing import List
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

def format_docs(docs):
    return "\n\n---\n\n".join([doc.page_content for doc in docs])