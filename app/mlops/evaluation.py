from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric
)
from deepeval.test_case import LLMTestCase

from app.mlops.deepeval_llm import DeepEvalOllamaLLM

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
    
    # Initialize Custom LLM
    try:
        ollama_llm = DeepEvalOllamaLLM()
        print(f"DEBUG: Initialized DeepEvalOllamaLLM with model {ollama_llm.get_model_name()}")
    except Exception as e:
        print(f"DEBUG: Failed to initialize DeepEvalOllamaLLM: {e}")
        raise

    # Calculate Answer Relevancy
    print(f"DEBUG: Calculating Answer Relevancy for query='{query[:50]}...'")
    answer_relevance = AnswerRelevancyMetric(model=ollama_llm)
    answer_relevance.measure(test_case)
    print(f"DEBUG: Answer Relevancy Score: {answer_relevance.score}")
    metrics["answer_relevance"] = answer_relevance.score

    # Calculate Faithfulness
    # Faithfulness also requires an LLM
    print("DEBUG: Calculating Faithfulness...")
    faithfulness = FaithfulnessMetric(model=ollama_llm)
    faithfulness.measure(test_case)
    print(f"DEBUG: Faithfulness Score: {faithfulness.score}")
    metrics["faithfulness"] = faithfulness.score

    # Calculate Contextual Precision
    # precision = ContextualPrecisionMetric(threshold=0.7)
    # precision.measure(test_case)
    # metrics["precision_at_k"] = precision.score

    # Calculate Contextual Recall
    # recall = ContextualRecallMetric()
    # recall.measure(test_case)
    # metrics["recall_at_k"] = recall.score

    return metrics
