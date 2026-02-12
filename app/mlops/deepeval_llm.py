# app/mlops/deepeval_llm.py

from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_ollama import ChatOllama
from app.config import settings


class DeepEvalOllamaLLM(DeepEvalBaseLLM):
    def __init__(self, model_name=None):
        self.model_name = model_name if model_name else settings.OLLAMA_MODEL
        self.llm = ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0,
            format="json" 
        )

    def load_model(self):
        return self.llm

    def generate(self, prompt: str) -> str:
        response = self.llm.invoke(prompt).content
        return self._clean_response(response)

    async def a_generate(self, prompt: str) -> str:
        res = await self.llm.ainvoke(prompt)
        response = res.content
        return self._clean_response(response)

    def _clean_response(self, response: str) -> str:
        """Nettoie la réponse pour enlever les blocs de code markdown si présents."""
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]
            
        if response.endswith("```"):
            response = response[:-3]
            
        response = response.strip()

        try:
            import json
            data = json.loads(response)
            
            if isinstance(data, dict):
                if "truths" not in data and "claims" in data:
                     data["truths"] = [] 
                
                if "truths" not in data:
                    data["truths"] = []
                
                if "claims" not in data:
                     data["claims"] = []
                     
                if "verdicts" not in data:
                    data["verdicts"] = []

                return json.dumps(data)
                
        except json.JSONDecodeError:
            pass
            
        return response



    def get_model_name(self):
        return self.model_name