"""
XAI LLM提供商适配器

实现了XAI API的调用与适配，使用langchain-xai库
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
from src.utils.llm_exception_utils import handle_llm_exception
from src.utils.async_utils import run_sync_in_executor

class XAIProvider(BaseLLMProvider):
    """XAI API适配器"""
    
    def __init__(self, api_key: str = None, api_base: str = None, **kwargs):
        """初始化XAI适配器
        
        Args:
            api_key: XAI API密钥
            api_base: 自定义API基础URL
            **kwargs: 其他参数
        """
        super().__init__(api_key, api_base, **kwargs)
        self._setup_client()
    
    def _setup_client(self):
        """设置XAI客户端"""
        try:
            self._validate_api_key()
            
            # 尝试导入langchain-xai
            try:
                from langchain_xai import ChatXAI
                self.chat_model_cls = ChatXAI
                self.use_langchain = True
            except ImportError:
                # 如果未安装则抛出异常
                raise ImportError("未安装langchain-xai库，请使用pip安装")
        except Exception as e:
            raise self._format_exception(e)
    
    async def generate_text(self, prompt: str, context: List[Dict], model: str,
                         temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """生成文本响应
        
        Args:
            prompt: 提示文本
            context: 上下文消息列表
            model: 模型名称，如xai-1-mini等
            temperature: 温度参数
            max_tokens: 最大生成令牌数
            
        Returns:
            str: 生成的文本
            
        Raises:
            LLMProviderException: XAI API调用异常
        """
        try:
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
                xai_api_key=self.api_key,
                xai_api_base=self.api_base if self.api_base else None,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            async def _generate():
                response = await chat_model.invoke(messages)
                return response.content

            return await handle_llm_exception(_generate)
        except Exception as e:
            raise self._format_exception(e)
    
    async def generate_embeddings(self, texts: List[str], model: str = None) -> List[List[float]]:
        """生成文本嵌入向量
        
        Args:
            texts: 文本列表
            model: 嵌入模型名称，默认为xai-embed-text-1
            
        Returns:
            List[List[float]]: 嵌入向量列表
            
        Raises:
            LLMProviderException: XAI API调用异常
        """
        try:
            if not model:
                model = "xai-embed-text-1"
                
            # 使用LangChain集成
            from langchain_xai import XAIEmbeddings
            
            embedding_model = XAIEmbeddings(
                model=model,
                xai_api_key=self.api_key,
                xai_api_base=self.api_base if self.api_base else None
            )
            
            async def _embed():
                return await embedding_model.embed_documents(texts)

            return await handle_llm_exception(_embed)
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
                model = "xai-1-mini"
            
            # XAI也使用cl100k_base编码器
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
            LLMProviderException: XAI API调用异常
        """
        try:
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
                xai_api_key=self.api_key,
                xai_api_base=self.api_base if self.api_base else None,
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
                "max_tokens": 768,  # XAI嵌入向量大小
            }
        
        # XAI聊天模型默认参数
        if "xai-1-mini" in model:
            return {
                "temperature": 0.7,
                "max_tokens": 1000,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }
        elif "xai-1" in model:
            return {
                "temperature": 0.7,
                "max_tokens": 2000,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }
        else:
            return super()._get_model_defaults(model, task_type)
