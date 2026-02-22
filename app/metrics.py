from prometheus_client import Counter, Histogram, Gauge

# RAG Metrics
RAG_REQUEST_TOTAL = Counter(
    "rag_request_total", 
    "Total number of RAG requests",
    ["status"]  # success, error
)

RAG_PROCESSING_TIME = Histogram(
    "rag_processing_seconds",
    "Time spent processing RAG requests",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

RAG_METRIC_FAITHFULNESS = Gauge(
    "rag_metric_faithfulness",
    "Faithfulness score from DeepEval",
    ["model"]
)

RAG_METRIC_ANSWER_RELEVANCE = Gauge(
    "rag_metric_answer_relevance",
    "Answer Relevance score from DeepEval",
    ["model"]
)

RAG_DOCS_RETRIEVED = Histogram(
    "rag_docs_retrieved_count",
    "Number of documents retrieved per query",
    buckets=[1, 3, 5, 10, 20]
)
