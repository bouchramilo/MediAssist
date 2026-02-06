import pytest
from textwrap import dedent
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document
from app.services.chunking import split_documents, estimate_tokens, split_by_paragraph, chunk_markdown_document

def test_estimate_tokens():
    text = "Hello world this is a test"
    assert estimate_tokens(text) == 6
    assert estimate_tokens("") == 0

def test_split_by_paragraph():
    text = "Para1 " * 100 + "\n\n" + "Para2 " * 100
    chunks = split_by_paragraph(text, max_tokens=50, overlap=10)
    assert len(chunks) > 1
    assert isinstance(chunks[0], str)

def test_chunk_markdown_document():
    text = dedent("""
    # Chapter 1
    
    ## Section 1
    Some content here.
    
    ## Section 2
    More content here.
    """).strip()
    
    chunks = chunk_markdown_document(text, source="test.pdf", page=1)
    
    assert len(chunks) == 3
    
    assert chunks[0]["metadata"]["chapter"] == "Chapter 1"
    assert chunks[0]["metadata"]["section"] is None
    
    assert chunks[1]["metadata"]["chapter"] == "Chapter 1"
    assert chunks[1]["metadata"]["section"] == "Section 1"
    
    assert chunks[2]["metadata"]["section"] == "Section 2"
    assert chunks[0]["metadata"]["chunk_type"] == "hierarchical"

def test_split_documents_success():
    doc = Document(
        page_content="# Title\n\n## Section 1\nContent.\n\n## Section 2\nContent.",
        metadata={"source": "doc1", "page": 5}
    )
    
    results = split_documents([doc])
    
    assert len(results) == 3
    assert isinstance(results[0], Document)
    assert results[0].metadata["source"] == "doc1"
    assert results[0].metadata["chapter"] == "Title"
    assert results[0].metadata["section"] is None
    
    assert results[1].metadata["section"] == "Section 1"

def test_split_documents_empty_input():
    result = split_documents([])
    assert result == []
