from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric
)
from deepeval.test_case import LLMTestCase

def evaluate_rag(query: str, response: str, context: list):
    """
    Evaluate RAG performance using DeepEval metrics.
    
    Args:
        query (str): The user query
        response (str): The generated response
        context (list): List of strings representing the retrieved context
    
    Returns:
        dict: calculated metrics
    """
    
    # Create test case for DeepEval
    test_case = LLMTestCase(
        input=query,
        actual_output=response,
        retrieval_context=context
    )

    metrics = {}

    # Calculate Answer Relevancy
    answer_relevance = AnswerRelevancyMetric()
    answer_relevance.measure(test_case)
    metrics["answer_relevance"] = answer_relevance.score

    # Calculate Faithfulness
    faithfulness = FaithfulnessMetric()
    faithfulness.measure(test_case)
    metrics["faithfulness"] = faithfulness.score

    # Calculate Contextual Precision
    precision = ContextualPrecisionMetric()
    precision.measure(test_case)
    metrics["precision_at_k"] = precision.score

    # Calculate Contextual Recall
    recall = ContextualRecallMetric()
    recall.measure(test_case)
    metrics["recall_at_k"] = recall.score

    return metrics
