# model_interfaces.py
from abc import ABC, abstractmethod
from data_models import ExperimentResult

class TtiApi(ABC):
    """Abstract Base Class (ABC) for any Text-to-Image API."""
    
    def __init__(self, model_name: str, provider: str):
        self.model_name = model_name
        self.provider = provider

    @abstractmethod
    async def generate_image(self, prompt: str) -> ExperimentResult:
        """
        The core method to generate an image. 
        MUST return a populated ExperimentResult object.
        """
        pass

# You can add a similar ABC for the LLM component if you plan to plug in 
# different LLM APIs (GPT-4o, Gemini Flash, etc.)