# app/mlops/evaluation.py

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from app.mlops.deepeval_llm import DeepEvalOllamaLLM
from app.utils.logger import AppLogger

logger = AppLogger.get_logger(__name__)

def evaluate_rag(query: str, response: str, context: list):
    """
    Evaluate RAG performance using DeepEval metrics.
    """
    metrics = {
        "answer_relevance": 0.0,
        "faithfulness": 0.0
    }
    
    try:
        # Create test case
        test_case = LLMTestCase(
            input=query,
            actual_output=response,
            retrieval_context=context
        )

        # Initialize LLM
        ollama_llm = DeepEvalOllamaLLM()
        
        # Calculate Answer Relevancy
        try:
            answer_relevance = AnswerRelevancyMetric(model=ollama_llm)
            answer_relevance.measure(test_case)
            metrics["answer_relevance"] = answer_relevance.score
            logger.info(f"Answer Relevancy: {answer_relevance.score}")
        except Exception as e:
            logger.warning(f"AnswerRelevancy failed: {e}")

        # Calculate Faithfulness
        try:
            faithfulness = FaithfulnessMetric(model=ollama_llm)
            faithfulness.measure(test_case)
            metrics["faithfulness"] = faithfulness.score
            logger.info(f"Faithfulness: {faithfulness.score}")
        except Exception as e:
            logger.warning(f"Faithfulness failed: {e}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
    
    return metrics