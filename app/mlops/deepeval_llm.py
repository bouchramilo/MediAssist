
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_ollama import ChatOllama
from app.config import settings

class DeepEvalOllamaLLM(DeepEvalBaseLLM):
    def __init__(self, model_name=None):
        self.model_name = model_name if model_name else settings.OLLAMA_MODEL
        self.llm = ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0,  # Evaluation should be deterministic
            format="json"   # Force JSON output for DeepEval
        )

    def load_model(self):
        return self.llm

    def generate(self, prompt: str) -> str:
        # DeepEval expects sync output for generate
        return self.llm.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        # DeepEval expects async output for a_generate
        res = await self.llm.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return self.model_name
