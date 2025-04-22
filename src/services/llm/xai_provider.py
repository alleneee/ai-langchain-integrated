# src/services/llm/xai_provider.py
"""
XAI Grok LLM Provider Adapter using langchain-xai.
"""
from src.services.llm.base import AbstractLLMProvider, register_provider
# Import the langchain strategy for XAI
from src.services.llm.strategies.xai_strategies import XAILangchainStrategy

@register_provider("xai") # Keeping registration name as "xai"
class XAIProvider(AbstractLLMProvider):
    """
    Concrete implementation of AbstractLLMProvider for xAI Grok model.
    Uses XAILangchainStrategy which utilizes the langchain-xai library.
    """

    def __init__(self, strategy: XAILangchainStrategy):
        """
        Initializes the XAIProvider with the XAI Langchain strategy.

        Args:
            strategy: An instance of XAILangchainStrategy.
        """
        # Ensure the correct strategy type is passed
        if not isinstance(strategy, XAILangchainStrategy):
             raise TypeError("XAIProvider requires an XAILangchainStrategy instance.")
        super().__init__(strategy)

    # Core methods (generate_text_async, generate_embeddings_async, etc.)
    # are inherited from AbstractLLMProvider and delegate to the strategy.