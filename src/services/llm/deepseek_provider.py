"""
DeepSeek LLM提供商适配器

实现了DeepSeek API的调用与适配，使用langchain-deepseek库
"""

import json
import tiktoken
import asyncio
from typing import Dict, List, Any, Optional, AsyncGenerator

from src.services.llm.base import BaseLLMProvider
from src.utils.llm_exception_utils import handle_llm_exception
from src.utils.async_utils import run_sync_in_executor

class DeepseekProvider(BaseLLMProvider):
    """DeepSeek API适配器"""
    
    def __init__(self, api_key: str = None, api_base: str = None, **kwargs):
        """初始化DeepSeek适配器
        
        Args:
            api_key: DeepSeek API密钥
            api_base: 自定义API基础URL
            **kwargs: 其他参数
        """
        super().__init__(api_key, api_base, **kwargs)
        try:
            self._setup_client()
        except Exception as e:
            handle_llm_exception(e, self.provider_name)
            self.client = None
            self.langchain_chat_model_cls = None
            self.langchain_embedding_model_cls = None
            self.use_langchain = False
            self.client_ready = False
        else:
            self.client_ready = True

    def _setup_client(self):
        """设置DeepSeek客户端 (同步部分)"""
        self._validate_api_key() # 这个本身会抛出AuthException，会被外部捕获
        
        self.langchain_chat_model_cls = None
        self.langchain_embedding_model_cls = None
        self.client = None
        self.use_langchain = False

        try:
            from langchain_deepseek import ChatDeepSeek, DeepseekEmbeddings
            self.langchain_chat_model_cls = ChatDeepSeek
            self.langchain_embedding_model_cls = DeepseekEmbeddings
            self.use_langchain = True
            print("DeepseekProvider: Using Langchain integration.")
        except ImportError:
            print("DeepseekProvider: langchain-deepseek not found, falling back to OpenAI compatible API.")
            try:
                import openai
                from openai import AsyncOpenAI
                self.client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base if self.api_base else "https://api.deepseek.com/v1",
                    timeout=60.0  # 增加超时
                )
                self.use_langchain = False
            except ImportError:
                raise ImportError("Neither langchain-deepseek nor openai library is installed. Please install one of them.")

    def _ensure_client_ready(self):
        if not self.client_ready:
            raise RuntimeError(f"{self.provider_name} client failed to initialize. Check configuration and logs.")
        
        pass

    async def generate_text(self, prompt: str, context: List[Dict], model: str,
                         temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """生成文本响应"""
        self._ensure_client_ready()
        try:
            if self.use_langchain and self.langchain_chat_model_cls:
                from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
                
                messages = []
                for msg in context:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user": messages.append(HumanMessage(content=content))
                    elif role == "assistant": messages.append(AIMessage(content=content))
                    elif role == "system": messages.append(SystemMessage(content=content))
                if prompt: messages.append(HumanMessage(content=prompt))
                
                chat_model = self.langchain_chat_model_cls(
                    model=model,
                    deepseek_api_key=self.api_key,
                    deepseek_api_base=self.api_base if self.api_base else None,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                response = await run_sync_in_executor(chat_model.invoke, messages)
                
                if hasattr(response, "content"): return response.content
                else: return str(response)
            elif not self.use_langchain and self.client:
                messages = self._prepare_messages(context, prompt)
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if response.choices and response.choices[0].message:
                    return response.choices[0].message.content
                else:
                    raise RuntimeError("DeepSeek API response format unexpected (no choices or message content)")
            else:
                raise RuntimeError(f"{self.provider_name} client not properly configured.")
        except Exception as e:
            handle_llm_exception(e, self.provider_name)
    
    async def generate_embeddings(self, texts: List[str], model: str = None) -> List[List[float]]:
        """生成文本嵌入向量"""
        self._ensure_client_ready()
        embedding_model_name = model if model else "deepseek-embedding" # DeepSeek 默认嵌入模型
        try:
            if self.use_langchain and self.langchain_embedding_model_cls:
                embedding_model = self.langchain_embedding_model_cls(
                    model=embedding_model_name,
                    deepseek_api_key=self.api_key,
                    deepseek_api_base=self.api_base if self.api_base else None
                )
                embeddings = await run_sync_in_executor(embedding_model.embed_documents, texts)
                return embeddings
            elif not self.use_langchain and self.client:
                response = await self.client.embeddings.create(
                    model=embedding_model_name,
                    input=texts
                )
                if response.data:
                    return [item.embedding for item in response.data]
                else:
                    raise RuntimeError("DeepSeek API embedding response format unexpected (no data)")
            else:
                raise RuntimeError(f"{self.provider_name} client not properly configured.")
        except Exception as e:
            handle_llm_exception(e, self.provider_name)

    async def count_tokens(self, text: str, model: str = None) -> Dict[str, int]:
        """计算文本的token数量"""
        try:
            model_name = model if model else "deepseek-chat" # 需要一个参考模型
            encoding = await run_sync_in_executor(tiktoken.encoding_for_model, model_name)
            tokens = await run_sync_in_executor(encoding.encode, text)
            token_count = len(tokens)
            char_count = len(text)
            return {"token_count": token_count, "char_count": char_count}
        except KeyError:
            try:
                encoding = await run_sync_in_executor(tiktoken.get_encoding, "cl100k_base")
                tokens = await run_sync_in_executor(encoding.encode, text)
                token_count = len(tokens)
                char_count = len(text)
                return {"token_count": token_count, "char_count": char_count}
            except Exception as e:
                handle_llm_exception(e, f"{self.provider_name} (tiktoken)")
        except Exception as e:
            handle_llm_exception(e, f"{self.provider_name} (tiktoken)")

    async def stream_chat(self, prompt: str, context: List[Dict], model: str,
                       temperature: float = 0.7, max_tokens: int = 1000) -> AsyncGenerator[str, None]:
        """流式生成聊天回复"""
        self._ensure_client_ready()
        try:
            if self.use_langchain and self.langchain_chat_model_cls:
                from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
                
                messages = []
                for msg in context:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user": messages.append(HumanMessage(content=content))
                    elif role == "assistant": messages.append(AIMessage(content=content))
                    elif role == "system": messages.append(SystemMessage(content=content))
                if prompt: messages.append(HumanMessage(content=prompt))
                
                chat_model = self.langchain_chat_model_cls(
                    model=model,
                    deepseek_api_key=self.api_key,
                    deepseek_api_base=self.api_base if self.api_base else None,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    streaming=True # 确保启用流式
                )
                
                async for chunk in chat_model.astream(messages):
                    if hasattr(chunk, "content"): yield chunk.content
                    elif isinstance(chunk, dict) and "content" in chunk: yield chunk["content"]
                    elif isinstance(chunk, str): yield chunk # 有些模型可能直接返回字符串块
                    # 忽略其他类型的块
            elif not self.use_langchain and self.client:
                messages = self._prepare_messages(context, prompt)
                
                stream = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True
                )
                
                async for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content
            else:
                raise RuntimeError(f"{self.provider_name} client not properly configured.")
        except Exception as e:
            handle_llm_exception(e, self.provider_name)
            
    def _get_model_defaults(self, model: str, task_type: str = "chat") -> Dict[str, Any]:
        """获取模型默认参数"""
        if task_type == "embedding":
            return {
                "max_tokens": 8192, # DeepSeek embedding model context length
            }
        
        if "deepseek-chat" in model:
            return {
                "temperature": 0.7,
                "max_tokens": 4096, # 参考模型能力调整
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }
        elif "deepseek" in model:
            return {
                "temperature": 0.7,
                "max_tokens": 2048,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }
        else:
            return super()._get_model_defaults(model, task_type)
