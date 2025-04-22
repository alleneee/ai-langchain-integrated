# src/services/llm/qwen_provider.py
"""
Qwen (DashScope) LLM Provider Adapter
"""
from src.services.llm.base import AbstractLLMProvider, register_provider
from src.services.llm.strategies.qwen_strategies import QwenLangchainStrategy # Import the strategy

@register_provider("qwen")
class QwenProvider(AbstractLLMProvider):
    """
    Concrete implementation of AbstractLLMProvider for Qwen/DashScope.
    Uses QwenLangchainStrategy for API interaction.
    """

    def __init__(self, strategy: QwenLangchainStrategy):
        """
        Initializes the QwenProvider with the Qwen strategy.

        Args:
            strategy: An instance of QwenLangchainStrategy.
        """
        if not isinstance(strategy, QwenLangchainStrategy):
             raise TypeError("QwenProvider requires a QwenLangchainStrategy instance.")
        super().__init__(strategy)

    # Core methods are inherited from AbstractLLMProvider and delegate to the strategy.