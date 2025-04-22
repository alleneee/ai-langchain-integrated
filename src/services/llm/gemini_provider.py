# src/services/llm/gemini_provider.py
"""
Google Gemini LLM Provider Adapter
"""
from src.services.llm.base import AbstractLLMProvider, register_provider
from src.services.llm.strategies.gemini_strategies import GeminiLangchainStrategy # Import the strategy

@register_provider("gemini")
class GeminiProvider(AbstractLLMProvider):
    """
    Concrete implementation of AbstractLLMProvider for Google Gemini.
    Uses GeminiLangchainStrategy for API interaction.
    """

    def __init__(self, strategy: GeminiLangchainStrategy):
        """
        Initializes the GeminiProvider with the Gemini strategy.

        Args:
            strategy: An instance of GeminiLangchainStrategy.
        """
        if not isinstance(strategy, GeminiLangchainStrategy):
             raise TypeError("GeminiProvider requires a GeminiLangchainStrategy instance.")
        super().__init__(strategy)

    # Core methods are inherited from AbstractLLMProvider and delegate to the strategy.