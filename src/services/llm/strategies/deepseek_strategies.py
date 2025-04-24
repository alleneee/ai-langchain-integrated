"""
LLM Interaction Strategy for Deepseek Provider.
"""

import os
import tiktoken
from typing import List, Dict, Any, Optional, AsyncGenerator

# Deepseek uses an OpenAI-compatible API for chat, but may need separate embedding handling.
# Assuming it uses OpenAIEmbeddings for now, consistent with original provider.
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.services.llm.base import LLMStrategy
from src.utils.llm_exception_utils import handle_llm_exception

_DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
_DEFAULT_DEEPSEEK_CHAT_MODEL = "deepseek-chat" # Or specify a default model
_DEFAULT_DEEPSEEK_EMBEDDING_MODEL = "text-embedding-ada-002" # Assuming use of OpenAI's for now

class DeepseekLangchainStrategy(LLMStrategy):
    """LLM Strategy implementation using Langchain for Deepseek."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = base_url or _DEEPSEEK_API_BASE
        self.extra_kwargs = kwargs

        if not self.api_key:
            raise ValueError("Deepseek API key is required for DeepseekLangchainStrategy.")

        try:
            # Use a common tokenizer, adjust if Deepseek requires a specific one
            self._tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo") 
        except KeyError:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")

    def _create_chat_client(self, model: str, temperature: float, max_tokens: int) -> ChatOpenAI:
        return ChatOpenAI(
            model=model or _DEFAULT_DEEPSEEK_CHAT_MODEL,
            openai_api_key=self.api_key,
            openai_api_base=self.base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            **(self.extra_kwargs.get("chat_options", {}))
        )

    def _create_embedding_client(self, model: Optional[str] = None) -> OpenAIEmbeddings:
        # Assuming Deepseek uses OpenAI-compatible embeddings or we fall back to OpenAI's service
        # This might need adjustment based on actual Deepseek embedding capabilities/API
        # If falling back to OpenAI, ensure OPENAI_API_KEY is also available or handled
        openai_api_key = os.getenv("OPENAI_API_KEY") # Or get from config if provided differently
        openai_base_url = os.getenv("OPENAI_API_BASE")
        embedding_model_name = model or _DEFAULT_DEEPSEEK_EMBEDDING_MODEL
        return OpenAIEmbeddings(
            model=embedding_model_name,
            api_key=openai_api_key, # Use OpenAI key if hitting OpenAI endpoint
            base_url=openai_base_url,
            **(self.extra_kwargs.get("embedding_options", {}))
        )

    def _prepare_langchain_messages(self, context: List[Dict], prompt: str) -> List:
        messages = []
        for msg in context:
            role = msg.get("role", "user").lower()
            content = msg.get("content", "")
            if content:
                if role == "user": messages.append(HumanMessage(content=content))
                elif role == "assistant" or role == "ai": messages.append(AIMessage(content=content))
                elif role == "system": messages.append(SystemMessage(content=content))
        if prompt: messages.append(HumanMessage(content=prompt))
        return messages

    @handle_llm_exception
    async def generate_text_async(self, prompt: str, context: List[Dict], model: str, temperature: float, max_tokens: int, **kwargs) -> str:
        llm = self._create_chat_client(model, temperature, max_tokens)
        messages = self._prepare_langchain_messages(context, prompt)
        response = await llm.ainvoke(messages, **kwargs)
        return response.content

    @handle_llm_exception
    async def generate_embeddings_async(self, texts: List[str], model: Optional[str] = None, **kwargs) -> List[List[float]]:
        embedder = self._create_embedding_client(model)
        return await embedder.aembed_documents(texts, **kwargs)

    @handle_llm_exception
    async def count_tokens_async(self, text: str, model: Optional[str] = None, **kwargs) -> Dict[str, int]:
        tokens = self._tokenizer.encode(text)
        return {"token_count": len(tokens), "character_count": len(text)}

    @handle_llm_exception
    async def stream_chat_async(self, prompt: str, context: List[Dict], model: str, temperature: float, max_tokens: int, **kwargs) -> AsyncGenerator[str, None]:
        llm = self._create_chat_client(model, temperature, max_tokens)
        messages = self._prepare_langchain_messages(context, prompt)
        async for chunk in llm.astream(messages, **kwargs):
            if chunk.content:
                 yield chunk.content

    def get_models(self) -> List[str]:
        # Provide known Deepseek models
        return ["deepseek-chat", "deepseek-coder"]
