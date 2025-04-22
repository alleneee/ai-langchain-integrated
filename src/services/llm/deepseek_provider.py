"""
DeepSeek LLM提供商适配器

实现了DeepSeek API的调用与适配，使用langchain-deepseek库
"""

import json
import tiktoken
from typing import Dict, List, Any, Optional, AsyncGenerator
from src.services.llm.base import BaseLLMProvider
from src.core.exceptions import (
    LLMProviderException, LLMProviderAuthException,
    LLMProviderModelNotFoundException
)
import asyncio

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
        self._setup_client()
    
    def _setup_client(self):
        """设置DeepSeek客户端"""
        try:
            self._validate_api_key()
            
            # 尝试导入langchain-deepseek
            try:
                from langchain_deepseek import ChatDeepSeek
                self.chat_model_cls = ChatDeepSeek
                self.use_langchain = True
            except ImportError:
                # 回退到直接使用OpenAI客户端，因为DeepSeek API兼容OpenAI
                import openai
                from openai import AsyncOpenAI
                self.client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base if self.api_base else "https://api.deepseek.com/v1",
                    timeout=30.0  # 设置30秒超时
                )
                self.use_langchain = False
        except Exception as e:
            raise self._format_exception(e)
    
    async def generate_text(self, prompt: str, context: List[Dict], model: str,
                         temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """生成文本响应
        
        Args:
            prompt: 提示文本
            context: 上下文消息列表
            model: 模型名称，如deepseek-chat等
            temperature: 温度参数
            max_tokens: 最大生成令牌数
            
        Returns:
            str: 生成的文本
            
        Raises:
            LLMProviderException: DeepSeek API调用异常
        """
        try:
            if self.use_langchain:
                # 使用LangChain集成
                from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
                
                # 转换为LangChain消息
                messages = []
                for msg in context:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    
                    if role == "user":
                        messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        messages.append(AIMessage(content=content))
                    elif role == "system":
                        messages.append(SystemMessage(content=content))
                
                # 添加当前提示
                if prompt:
                    messages.append(HumanMessage(content=prompt))
                
                # 创建LangChain模型
                chat_model = self.chat_model_cls(
                    model=model,
                    deepseek_api_key=self.api_key,
                    deepseek_api_base=self.api_base if self.api_base else None,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # 调用模型
                response = await asyncio.to_thread(chat_model.invoke, messages)
                
                # 提取响应内容
                if hasattr(response, "content"):
                    return response.content
                else:
                    return str(response)
            else:
                # 使用OpenAI客户端
                messages = self._prepare_messages(context, prompt)
                
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                if response.choices and len(response.choices) > 0:
                    return response.choices[0].message.content
                
                return ""
        except Exception as e:
            raise self._format_exception(e)
    
    async def generate_embeddings(self, texts: List[str], model: str = None) -> List[List[float]]:
        """生成文本嵌入向量
        
        Args:
            texts: 文本列表
            model: 嵌入模型名称，默认为deepseek-embedding
            
        Returns:
            List[List[float]]: 嵌入向量列表
            
        Raises:
            LLMProviderException: DeepSeek API调用异常
        """
        try:
            if not model:
                model = "deepseek-embedding"
                
            if self.use_langchain:
                # 使用LangChain集成
                from langchain_deepseek import DeepSeekEmbeddings
                
                embedding_model = DeepSeekEmbeddings(
                    model=model,
                    deepseek_api_key=self.api_key,
                    deepseek_api_base=self.api_base if self.api_base else None
                )
                
                # 使用LangChain嵌入
                embeddings = await asyncio.to_thread(embedding_model.embed_documents, texts)
                return embeddings
            else:
                # 使用OpenAI客户端
                response = await self.client.embeddings.create(
                    model=model,
                    input=texts
                )
                
                embeddings = [data.embedding for data in response.data]
                return embeddings
        except Exception as e:
            raise self._format_exception(e)
    
    async def count_tokens(self, text: str, model: str = None) -> Dict[str, int]:
        """计算文本的token数量
        
        Args:
            text: 待计算的文本
            model: 模型名称
            
        Returns:
            Dict[str, int]: 包含token数量和字符数量的字典
            
        Raises:
            LLMProviderException: 计算token异常
        """
        try:
            if not model:
                model = "deepseek-chat"
            
            # DeepSeek模型使用cl100k_base编码器
            encoding = tiktoken.get_encoding("cl100k_base")
            
            # 计算token
            tokens = encoding.encode(text)
            token_count = len(tokens)
            
            return {
                "token_count": token_count,
                "character_count": len(text)
            }
        except Exception as e:
            # 如果tiktoken出现问题，使用近似计算
            return {
                "token_count": len(text) // 4,  # 粗略估计每个token约4个字符
                "character_count": len(text)
            }
    
    async def stream_chat(self, prompt: str, context: List[Dict], model: str,
                       temperature: float = 0.7, max_tokens: int = 1000) -> AsyncGenerator[str, None]:
        """流式生成聊天回复
        
        Args:
            prompt: 提示文本
            context: 上下文消息列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大生成令牌数
            
        Yields:
            str: 生成的文本片段
            
        Raises:
            LLMProviderException: DeepSeek API调用异常
        """
        try:
            if self.use_langchain:
                # 使用LangChain集成
                from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
                
                # 转换为LangChain消息
                messages = []
                for msg in context:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    
                    if role == "user":
                        messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        messages.append(AIMessage(content=content))
                    elif role == "system":
                        messages.append(SystemMessage(content=content))
                
                # 添加当前提示
                if prompt:
                    messages.append(HumanMessage(content=prompt))
                
                # 创建LangChain流式模型
                chat_model = self.chat_model_cls(
                    model=model,
                    deepseek_api_key=self.api_key,
                    deepseek_api_base=self.api_base if self.api_base else None,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    streaming=True
                )
                
                # 调用流式响应
                async for chunk in await asyncio.to_thread(chat_model.astream, messages):
                    if hasattr(chunk, "content"):
                        yield chunk.content
                    elif isinstance(chunk, dict) and "content" in chunk:
                        yield chunk["content"]
                    else:
                        yield str(chunk)
            else:
                # 使用OpenAI客户端
                messages = self._prepare_messages(context, prompt)
                
                stream = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True
                )
                
                async for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
        except Exception as e:
            raise self._format_exception(e)
            
    def _get_model_defaults(self, model: str, task_type: str = "chat") -> Dict[str, Any]:
        """获取模型默认参数
        
        Args:
            model: 模型名称
            task_type: 任务类型
            
        Returns:
            Dict[str, Any]: 模型默认参数
        """
        if task_type == "embedding":
            return {
                "max_tokens": 8192,
            }
        
        # DeepSeek聊天模型默认参数
        if "deepseek-chat" in model:
            return {
                "temperature": 0.7,
                "max_tokens": 2000,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }
        # 其他DeepSeek模型
        elif "deepseek" in model:
            return {
                "temperature": 0.7,
                "max_tokens": 1000,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }
        else:
            return super()._get_model_defaults(model, task_type)
