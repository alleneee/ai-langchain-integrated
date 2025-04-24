"""
LLM Interaction Strategies for OpenAI Provider.
"""

import os
import tiktoken # OpenAI's tokenizer
from typing import List, Dict, Any, Optional, AsyncGenerator, Callable

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool

from src.services.llm.base import LLMStrategy
from src.utils.llm_exception_utils import handle_llm_exception

# Consider centralizing model defaults
_DEFAULT_OPENAI_CHAT_MODEL = "gpt-3.5-turbo"
_DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"

class OpenAILangchainStrategy(LLMStrategy):
    """LLM Strategy implementation using Langchain for OpenAI."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_API_BASE")
        self.extra_kwargs = kwargs # Store extra config if needed (e.g., specific model params)

        if not self.api_key:
            raise ValueError("OpenAI API key is required for OpenAILangchainStrategy.")
        
        # Pre-load tokenizer encoding based on a common model
        # This might need adjustment if supporting vastly different models
        try:
            self._tokenizer = tiktoken.encoding_for_model(_DEFAULT_OPENAI_CHAT_MODEL)
        except KeyError:
            print(f"Warning: Could not get tokenizer for {_DEFAULT_OPENAI_CHAT_MODEL}. Using default cl100k_base.")
            self._tokenizer = tiktoken.get_encoding("cl100k_base")

    def _create_chat_client(self, model: str, temperature: float, max_tokens: int) -> ChatOpenAI:
        """Helper to create a ChatOpenAI client."""
        return ChatOpenAI(
            model=model or _DEFAULT_OPENAI_CHAT_MODEL,
            api_key=self.api_key,  # 更新为新的参数名
            base_url=self.base_url,  # 更新为新的参数名
            temperature=temperature,
            max_tokens=max_tokens,
            **(self.extra_kwargs.get("chat_options", {}))
        )

    def _create_embedding_client(self, model: Optional[str] = None) -> OpenAIEmbeddings:
        """Helper to create an OpenAIEmbeddings client."""
        embedding_model_name = model or _DEFAULT_OPENAI_EMBEDDING_MODEL
        return OpenAIEmbeddings(
            model=embedding_model_name,
            api_key=self.api_key,  # 更新为新的参数名
            base_url=self.base_url,  # 更新为新的参数名
            **(self.extra_kwargs.get("embedding_options", {}))
        )

    def _prepare_langchain_messages(self, context: List[Dict], prompt: str) -> List:
        """Converts context and prompt into LangChain message objects."""
        messages = []
        for msg in context:
            role = msg.get("role", "user").lower()
            content = msg.get("content", "")
            
            # 处理可能包含的工具调用响应
            tool_calls = msg.get("tool_calls", [])
            tool_call_id = msg.get("tool_call_id")
            
            if content:
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant" or role == "ai":
                    # 处理可能的工具调用
                    if tool_calls:
                        # 创建带有工具调用的AI消息
                        messages.append(AIMessage(content=content, tool_calls=tool_calls))
                    else:
                        messages.append(AIMessage(content=content))
                elif role == "system":
                    messages.append(SystemMessage(content=content))
                elif role == "tool" and tool_call_id:
                    # 添加工具消息支持
                    messages.append(ToolMessage(content=content, tool_call_id=tool_call_id))
                    
        if prompt:
            messages.append(HumanMessage(content=prompt))
        return messages

    @handle_llm_exception
    async def generate_text_async(self, prompt: str, context: List[Dict], model: str,
                               temperature: float, max_tokens: int, **kwargs) -> str:
        llm = self._create_chat_client(model, temperature, max_tokens)
        messages = self._prepare_langchain_messages(context, prompt)
        response = await llm.ainvoke(messages, **kwargs) # Pass additional kwargs if needed
        return response.content

    @handle_llm_exception
    async def generate_embeddings_async(self, texts: List[str], model: Optional[str] = None, **kwargs) -> List[List[float]]:
        embedder = self._create_embedding_client(model)
        return await embedder.aembed_documents(texts, **kwargs)

    @handle_llm_exception
    async def count_tokens_async(self, text: str, model: Optional[str] = None, **kwargs) -> Dict[str, int]:
        """Counts tokens using tiktoken."""
        # Note: model parameter is currently unused here as we pre-loaded tokenizer.
        # If different models need different tokenizers, this needs refinement.
        tokens = self._tokenizer.encode(text)
        token_count = len(tokens)
        character_count = len(text)
        return {"token_count": token_count, "character_count": character_count}

    @handle_llm_exception
    async def stream_chat_async(self, prompt: str, context: List[Dict], model: str,
                             temperature: float, max_tokens: int, **kwargs) -> AsyncGenerator[str, None]:
        llm = self._create_chat_client(model, temperature, max_tokens)
        messages = self._prepare_langchain_messages(context, prompt)
        async for chunk in llm.astream(messages, **kwargs):
            if chunk.content:
                 yield chunk.content
                 
    @handle_llm_exception
    async def chat_with_tools(self, prompt: str, context: List[Dict], tools: List[BaseTool], 
                           model: str, temperature: float, max_tokens: int, **kwargs) -> Dict[str, Any]:
        """使用工具进行聊天，符合LangChain最新工具调用模式。"""
        llm = self._create_chat_client(model, temperature, max_tokens)
        messages = self._prepare_langchain_messages(context, prompt)
        
        # 绑定工具到模型
        llm_with_tools = llm.bind_tools(tools)
        
        # 调用模型生成响应
        response = await llm_with_tools.ainvoke(messages, **kwargs)
        
        return {
            "content": response.content,
            "tool_calls": getattr(response, "tool_calls", None)
        }

    def get_models(self) -> List[str]:
        """Returns a static list of common OpenAI models."""
        # TODO: Potentially fetch dynamically from OpenAI API
        return [
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-4-turbo-preview", # Often an alias
            "gpt-4",
            "gpt-3.5-turbo",
            "text-embedding-3-large",
            "text-embedding-3-small",
            "text-embedding-ada-002",
        ]
