# src/services/llm/llm_factory.py

from typing import Dict, Any, Optional
import os

from src.services.llm.base import AbstractLLMProvider, _provider_registry, LLMStrategy # Import necessary base components
from src.services.llm.strategies.openai_strategies import OpenAILangchainStrategy
from src.services.llm.strategies.gemini_strategies import GeminiLangchainStrategy
from src.services.llm.strategies.xai_strategies import XAILangchainStrategy # Import the langchain strategy
# Import other strategies as they are created
# from src.services.llm.strategies.deepseek_strategies import DeepseekLangchainStrategy
# from src.services.llm.strategies.qwen_strategies import QwenLangchainStrategy
# ... etc.

class LLMFactory:
    """
    Factory class for creating LLM Provider instances with appropriate strategies.
    """

    @staticmethod
    def create_strategy(provider_name: str, config: Dict[str, Any]) -> LLMStrategy:
        """
        Creates the appropriate LLMStrategy instance based on the provider name and config.

        Args:
            provider_name: The name of the provider (e.g., 'openai', 'deepseek').
            config: A dictionary containing configuration required for the strategy
                    (e.g., api_key, base_url, model defaults, strategy_options).

        Returns:
            An instance of the corresponding LLMStrategy.

        Raises:
            ValueError: If the provider name is unknown or configuration is missing/invalid.
        """
        provider_name = provider_name.lower()
        # Extract common potential config keys safely
        api_key = config.get('api_key') # Let strategy handle env var fallback if config key is missing
        base_url = config.get('base_url')
        strategy_kwargs = config.get('strategy_options', {})

        if provider_name == "openai":
            return OpenAILangchainStrategy(api_key=api_key, base_url=base_url, **strategy_kwargs)
        elif provider_name == "deepseek":
            # Deepseek strategy handles its own DEEPSEEK_API_KEY env var if api_key is None
            return DeepseekLangchainStrategy(api_key=api_key, base_url=base_url, **strategy_kwargs)
        elif provider_name == "qwen":
            # Qwen strategy handles DASHSCOPE_API_KEY env var if api_key is None
            return QwenLangchainStrategy(api_key=api_key, base_url=base_url, **strategy_kwargs)
        elif provider_name == "gemini":
            # Gemini strategy handles GOOGLE_API_KEY env var if api_key is None
            # Gemini doesn't typically use base_url in the same way
            return GeminiLangchainStrategy(api_key=api_key, **strategy_kwargs)
        elif provider_name == "xai":
            # Use the Langchain strategy for XAI
            # XAILangchainStrategy handles XAI_API_KEY env var if api_key is None
            return XAILangchainStrategy(
                api_key=api_key,        # Pass api_key from config (or None)
                api_base=base_url,      # Pass base_url as api_base (or None)
                **strategy_kwargs       # Pass strategy-specific options
            )
        else:
            raise ValueError(f"Unknown or unsupported LLM provider strategy: {provider_name}")

    @staticmethod
    def create_provider(provider_name: str, config: Optional[Dict[str, Any]] = None) -> AbstractLLMProvider:
        """
        Creates an LLM Provider instance for the given name, injecting the appropriate strategy.

        Args:
            provider_name: The registered name of the provider (e.g., 'openai').
            config: Configuration dictionary passed to the strategy constructor.
                    If None, defaults to an empty dict, relying on environment variables within the strategy.

        Returns:
            An instance of the requested AbstractLLMProvider subclass.

        Raises:
            ValueError: If the provider name is not registered or strategy creation fails.
        """
        provider_name = provider_name.lower()
        if provider_name not in _provider_registry:
            raise ValueError(f"LLM provider '{provider_name}' is not registered.")

        provider_class = _provider_registry[provider_name]

        # Ensure config is a dictionary for strategy creation
        effective_config = config if config is not None else {}

        # Create the strategy using the dedicated method
        strategy = LLMFactory.create_strategy(provider_name, effective_config)

        # Instantiate the provider with the created strategy
        # The provider's __init__ now expects only the strategy
        provider_instance = provider_class(strategy=strategy)

        return provider_instance

# Example Usage (how client code would use the factory):
# if __name__ == "__main__":
#     try:
#         # Config could come from settings, API request, etc.
#         openai_config = {
#             # api_key and base_url could be omitted if set as environment variables
#             # 'api_key': 'YOUR_OPENAI_KEY',
#             # 'base_url': 'YOUR_OPENAI_BASE_URL',
#             'strategy_options': { # Optional extra args for the strategy's clients
#                  'chat_options': {'timeout': 30},
#                  'embedding_options': {}
#              }
#         }
#         openai_provider = LLMFactory.create_provider("openai", config=openai_config)
#         print(f"Successfully created provider: {type(openai_provider).__name__} with strategy: {type(openai_provider.strategy).__name__}")
#
#         # Now you can use the provider methods
#         # models = openai_provider.get_models()
#         # print("Available models:", models)
#
#     except ValueError as e:
#         print(f"Error creating provider: {e}")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")

#         # Example: Create XAI/Groq provider (config can be empty if key is in env)
#         # Ensure XAI_API_KEY is set in environment
#         xai_config = {'strategy_options': {'chat_options': {'temperature': 0.5}}}
#         xai_provider = LLMFactory.create_provider("xai", config=xai_config)
#         print(f"Created provider: {type(xai_provider).__name__} with strategy: {type(xai_provider.strategy).__name__}")
#         # models = xai_provider.get_models()
