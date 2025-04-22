"""
LLM Provider Base Module

Defines the abstract base classes and interfaces for LLM providers and strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, AsyncGenerator, Type

# Assuming LLMProviderInterface defines the core methods expected by clients
# If it doesn't exist or needs update, adjust accordingly.
# Let's assume it exists for now.
from src.core.interfaces import LLMProviderInterface
from src.core.exceptions import (
    LLMProviderException, LLMProviderAuthException,
    LLMProviderQuotaException, LLMProviderRateLimitException,
    LLMProviderModelNotFoundException
)

# +--------------------------------------------------------------------+
# | LLM Interaction Strategy Interface                                 |
# +--------------------------------------------------------------------+

class LLMStrategy(ABC):
    """Abstract base class for LLM interaction strategies (Langchain, Native API, etc.)."""

    @abstractmethod
    async def generate_text_async(self, prompt: str, context: List[Dict], model: str,
                               temperature: float, max_tokens: int, **kwargs) -> str:
        """Generates a text response asynchronously."""
        pass

    @abstractmethod
    async def generate_embeddings_async(self, texts: List[str], model: Optional[str] = None, **kwargs) -> List[List[float]]:
        """Generates text embeddings asynchronously."""
        pass

    @abstractmethod
    async def count_tokens_async(self, text: str, model: Optional[str] = None, **kwargs) -> Dict[str, int]:
        """Counts tokens in the text asynchronously."""
        pass

    @abstractmethod
    async def stream_chat_async(self, prompt: str, context: List[Dict], model: str,
                             temperature: float, max_tokens: int, **kwargs) -> AsyncGenerator[str, None]:
        """Streams chat responses asynchronously."""
        # Required for abstract async generator
        if False:
            yield
        pass

    @abstractmethod
    def get_models(self) -> List[str]:
        """Gets the list of supported models for this strategy/provider."""
        pass

# +--------------------------------------------------------------------+
# | Abstract LLM Provider                                              |
# +--------------------------------------------------------------------+

# Rename BaseLLMProvider to AbstractLLMProvider
class AbstractLLMProvider(LLMProviderInterface, ABC):
    """
    Abstract base class for LLM Providers.
    Delegates core LLM operations to an injected LLMStrategy.
    Ensures a consistent interface for clients.
    """
    strategy: LLMStrategy
    provider_name: str

    def __init__(self, strategy: LLMStrategy):
        """
        Initializes the provider with a specific interaction strategy.

        Args:
            strategy: The LLMStrategy instance to use for LLM operations.
        """
        if not isinstance(strategy, LLMStrategy):
            raise TypeError("strategy must be an instance of LLMStrategy")
        self.strategy = strategy
        # Derive provider name from class name by default
        # Can be overridden by concrete class or factory if needed
        self.provider_name = self.__class__.__name__.replace("Provider", "")

    async def generate_text(self, prompt: str, context: List[Dict], model: str,
                         temperature: float = 0.7, max_tokens: int = 1000, **kwargs) -> str:
        """Generates a text response by delegating to the strategy."""
        return await self.strategy.generate_text_async(
            prompt=prompt, context=context, model=model,
            temperature=temperature, max_tokens=max_tokens, **kwargs
        )

    async def generate_embeddings(self, texts: List[str], model: Optional[str] = None, **kwargs) -> List[List[float]]:
        """Generates text embeddings by delegating to the strategy."""
        return await self.strategy.generate_embeddings_async(
            texts=texts, model=model, **kwargs
        )

    async def count_tokens(self, text: str, model: Optional[str] = None, **kwargs) -> Dict[str, int]:
        """Counts tokens by delegating to the strategy."""
        # Note: The original base class had an abstract count_tokens.
        # The strategy now handles this. Ensure the interface matches.
        return await self.strategy.count_tokens_async(
            text=text, model=model, **kwargs
        )

    async def stream_chat(self, prompt: str, context: List[Dict], model: str,
                       temperature: float = 0.7, max_tokens: int = 1000, **kwargs) -> AsyncGenerator[str, None]:
        """Streams chat responses by delegating to the strategy."""
        async for chunk in self.strategy.stream_chat_async(
            prompt=prompt, context=context, model=model,
            temperature=temperature, max_tokens=max_tokens, **kwargs
        ):
            yield chunk

    def get_models(self) -> List[str]:
        """Gets the list of supported models by delegating to the strategy."""
        return self.strategy.get_models()

    # --- Utility methods (can be kept if generic enough, or moved to utils) ---
    # These methods might be better placed in a utility module or within specific strategies
    # if they are not universally applicable or needed by the provider itself.

    def _prepare_messages(self, context: List[Dict], prompt: str = None) -> List[Dict[str, str]]:
        """
        Helper to convert context and prompt into a standard message format.
        This might be useful for strategies, or strategies might have their own formatters.
        Consider moving to a dedicated utility module if used across multiple strategies.
        """
        messages = []
        if context:
            for msg in context:
                role = msg.get("role", "user") # Default to user if role is missing
                content = msg.get("content", "")
                # Basic validation: only add if role and content are present
                if role and content:
                    # Standardize common roles if needed (e.g., 'assistant' vs 'ai')
                    standard_role = role.lower()
                    messages.append({"role": standard_role, "content": content})

        # Add the current prompt as the last user message
        if prompt:
             messages.append({"role": "user", "content": prompt})

        return messages

    # Methods like _validate_api_key and _get_model_defaults are strategy-specific
    # and should be implemented within the concrete strategy classes, not here.

# +--------------------------------------------------------------------+
# | Provider Registration (Keep if used by factory)                    |
# +--------------------------------------------------------------------+

_providers: Dict[str, Type[AbstractLLMProvider]] = {}

def register_provider(name: str):
    """Decorator to register LLM provider classes."""
    def decorator(cls):
        if not issubclass(cls, AbstractLLMProvider):
             raise TypeError(f"{cls.__name__} must inherit from AbstractLLMProvider")
        _providers[name] = cls
        # print(f"Registered provider: {name} -> {cls.__name__}") # Optional: for debugging
        return cls
    return decorator

def get_provider_cls(name: str) -> Type[AbstractLLMProvider]:
    """Gets the registered provider class by name."""
    cls = _providers.get(name)
    if cls is None:
        # Consider trying to dynamically import if not found?
        # Example: Dynamically import src.services.llm.{name}_provider.{ProviderName}
        raise ValueError(f"Unknown or unregistered provider: {name}. Available: {list(_providers.keys())}")
    return cls

def get_available_providers() -> List[str]:
    """Returns a list of names of registered providers."""
    return list(_providers.keys())
