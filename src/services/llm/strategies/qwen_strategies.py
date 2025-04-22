"""
LLM Interaction Strategy for Qwen (DashScope) Provider.
"""

import os
import tiktoken
from typing import List, Dict, Any, Optional, AsyncGenerator

# Qwen uses DashScope, which has its own Langchain integrations
# Chat might be OpenAI compatible, but embeddings are specific.
from langchain_openai import ChatOpenAI # Assuming compatibility for chat
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.services.llm.base import LLMStrategy
from src.utils.llm_exception_utils import handle_llm_exception

# Qwen often uses DASHSCOPE_API_KEY
_QWEN_API_KEY_NAME = "DASHSCOPE_API_KEY"
_QWEN_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1" # Example, verify correct endpoint
_DEFAULT_QWEN_CHAT_MODEL = "qwen-turbo" # Example default
_DEFAULT_QWEN_EMBEDDING_MODEL = "text-embedding-v1" # Example default

class QwenLangchainStrategy(LLMStrategy):
    """LLM Strategy implementation using Langchain for Qwen/DashScope."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        # Qwen uses DASHSCOPE_API_KEY
        self.api_key = api_key or os.getenv(_QWEN_API_KEY_NAME)
        # Base URL might be needed for ChatOpenAI if using compatible mode
        self.base_url = base_url or _QWEN_API_BASE 
        self.extra_kwargs = kwargs

        if not self.api_key:
            raise ValueError(f"{_QWEN_API_KEY_NAME} is required for QwenLangchainStrategy.")

        try:
            self._tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo") # Use appropriate tokenizer
        except KeyError:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")

    def _create_chat_client(self, model: str, temperature: float, max_tokens: int) -> ChatOpenAI:
        # Assuming Qwen provides an OpenAI-compatible endpoint for chat
        return ChatOpenAI(
            model=model or _DEFAULT_QWEN_CHAT_MODEL,
            openai_api_key=self.api_key,
            openai_api_base=self.base_url, # Use compatible base URL
            temperature=temperature,
            max_tokens=max_tokens, # Note: Qwen might handle max_tokens differently
            **(self.extra_kwargs.get("chat_options", {}))
        )

    def _create_embedding_client(self, model: Optional[str] = None) -> DashScopeEmbeddings:
        # Use DashScopeEmbeddings for Qwen
        embedding_model_name = model or _DEFAULT_QWEN_EMBEDDING_MODEL
        return DashScopeEmbeddings(
            model=embedding_model_name,
            dashscope_api_key=self.api_key,
            **(self.extra_kwargs.get("embedding_options", {}))
        )

    def _prepare_langchain_messages(self, context: List[Dict], prompt: str) -> List:
        # Same message preparation logic as OpenAI compatible
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
        # Qwen might have its own tokenizer or standard way to count.
        # Using tiktoken as a fallback/approximation.
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
        # Provide known Qwen/DashScope models
        return [
            "qwen-turbo", "qwen-plus", "qwen-max", "qwen-max-longcontext",
            "text-embedding-v1", "text-embedding-v2"
            # Add other models as needed
        ]
