# src/services/llm/strategies/xai_strategies.py
"""
LLM Interaction Strategy for XAI Grok Provider using langchain-xai.
"""

import os
import tiktoken
from typing import List, Dict, Any, Optional, AsyncGenerator

# Import components from langchain-xai
try:
    from langchain_xai import ChatXAI, XAIEmbeddings
except ImportError:
    raise ImportError("langchain-xai package not found. Please install it using 'pip install langchain-xai==0.2.3'")

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.services.llm.base import LLMStrategy
from src.utils.llm_exception_utils import handle_llm_exception
from src.utils.llm_message_utils import prepare_langchain_messages

# Constants based on langchain-xai usage
_XAI_API_KEY_NAME = "XAI_API_KEY"
_DEFAULT_XAI_CHAT_MODEL = "grok-1" # Default chat model for xAI
_DEFAULT_XAI_EMBEDDING_MODEL = "text-embedding-ada-002" # Placeholder - Verify if XAIEmbeddings needs a model param or has a default

class XAILangchainStrategy(LLMStrategy):
    """LLM Strategy implementation using langchain-xai for XAI Grok."""

    def __init__(self, api_key: Optional[str] = None, api_base: Optional[str] = None, **kwargs):
        """
        Initializes the XAILangchainStrategy.

        Args:
            api_key: XAI API Key. Falls back to XAI_API_KEY environment variable.
            api_base: Optional XAI API base URL.
            **kwargs: Additional keyword arguments for underlying clients.
        """
        self.api_key = api_key or os.getenv(_XAI_API_KEY_NAME)
        self.api_base = api_base # Optional base URL
        self.extra_kwargs = kwargs

        if not self.api_key:
            raise ValueError(f"{_XAI_API_KEY_NAME} is required for XAILangchainStrategy.")

        # Tokenizer for counting (assuming cl100k_base similar to Groq/OpenAI)
        # Check langchain-xai docs if they provide specific tokenizer info
        try:
             # Using a common encoder as proxy if specific one isn't known
             self._tokenizer = tiktoken.encoding_for_model("gpt-4")
        except KeyError:
             self._tokenizer = tiktoken.get_encoding("cl100k_base")


    def _create_chat_client(self, model: str, temperature: float, max_tokens: int, streaming: bool = False) -> ChatXAI:
        """Creates a ChatXAI client instance."""
        # Check langchain-xai constructor for correct parameters
        # Assumes it takes model, api_key, api_base, temperature, max_tokens, streaming
        return ChatXAI(
            model=model or _DEFAULT_XAI_CHAT_MODEL,
            xai_api_key=self.api_key,
            xai_api_base=self.api_base if self.api_base else None,
            temperature=temperature,
            max_tokens=max_tokens, # Verify parameter name in langchain-xai
            streaming=streaming,
            **(self.extra_kwargs.get("chat_options", {})) # Pass additional options
        )

    def _create_embedding_client(self, model: Optional[str] = None) -> XAIEmbeddings:
        """Creates an XAIEmbeddings client instance."""
        # Check langchain-xai constructor for parameters.
        # Does it take 'model'? What is the default?
        embedding_model_name = model # Pass model if specified, otherwise rely on langchain-xai default
        return XAIEmbeddings(
            model=embedding_model_name, # <<<--- VERIFY if 'model' is the correct param name
            xai_api_key=self.api_key,
            xai_api_base=self.api_base if self.api_base else None,
            **(self.extra_kwargs.get("embedding_options", {})) # Pass additional options
        )

    @handle_llm_exception
    async def generate_text_async(self, prompt: str, context: List[Dict], model: str, temperature: float, max_tokens: int, **kwargs) -> str:
        """Generates text using ChatXAI."""
        llm = self._create_chat_client(model, temperature, max_tokens, streaming=False)
        messages = prepare_langchain_messages(context, prompt)
        response = await llm.ainvoke(messages, **kwargs)
        return response.content

    @handle_llm_exception
    async def generate_embeddings_async(self, texts: List[str], model: Optional[str] = None, **kwargs) -> List[List[float]]:
        """Generates embeddings using XAIEmbeddings."""
        # Verify if XAIEmbeddings exists and works as expected
        embedder = self._create_embedding_client(model)
        return await embedder.aembed_documents(texts, **kwargs)

    @handle_llm_exception
    async def count_tokens_async(self, text: str, model: Optional[str] = None, **kwargs) -> Dict[str, int]:
        """Counts tokens using tiktoken approximation."""
        # Revisit if langchain-xai provides a specific token counting method
        print("Warning: Using tiktoken for xAI token count approximation.")
        tokens = self._tokenizer.encode(text)
        return {"token_count": len(tokens), "character_count": len(text)}

    @handle_llm_exception
    async def stream_chat_async(self, prompt: str, context: List[Dict], model: str, temperature: float, max_tokens: int, **kwargs) -> AsyncGenerator[str, None]:
        """Streams chat responses using ChatXAI."""
        llm = self._create_chat_client(model, temperature, max_tokens, streaming=True)
        messages = prepare_langchain_messages(context, prompt)
        async for chunk in llm.astream(messages, **kwargs):
            if chunk.content:
                 yield chunk.content

    def get_models(self) -> List[str]:
        """Returns known xAI models."""
        # <<< --- VERIFY ACTUAL MODEL NAMES SUPPORTED BY langchain-xai ---
        chat_models = ["grok-1"] # Add others if known (e.g., grok-1.5)
        # <<< --- VERIFY EMBEDDING MODEL NAMES ---
        # Check if XAIEmbeddings requires/supports specific model names
        embedding_models = [_DEFAULT_XAI_EMBEDDING_MODEL] # Placeholder
        return chat_models + embedding_models
