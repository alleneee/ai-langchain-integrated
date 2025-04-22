"""
OpenAI LLM提供商适配器

实现了OpenAI API的调用与适配
"""

import json
import tiktoken
from typing import Dict, List, Any, Optional, AsyncGenerator
import openai
from openai import AsyncOpenAI
from src.services.llm.base import BaseLLMProvider
from src.core.exceptions import (
    LLMProviderException, LLMProviderAuthException,
    LLMProviderModelNotFoundException
)
from src.utils.llm_utils import handle_llm_exception

class OpenAIProvider(BaseLLMProvider):
    """OpenAI API适配器"""
    
    def __init__(self, api_key: str = None, api_base: str = None, organization: str = None, **kwargs):
        """初始化OpenAI适配器
        
        Args:
            api_key: OpenAI API密钥
            api_base: 自定义API基础URL
            organization: 组织ID
            **kwargs: 其他参数
        """
        super().__init__(api_key, api_base, **kwargs)
        self.organization = organization
        self._setup_client()
    
    def _setup_client(self):
        """设置OpenAI客户端"""
        try:
            self._validate_api_key()
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.api_base if self.api_base else None,
                organization=self.organization if self.organization else None,
                timeout=30.0  # 设置30秒超时
            )
        except Exception as e:
            raise self._format_exception(e)
    
    async def generate_text(self, prompt: str, context: List[Dict], model: str,
                         temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """生成文本响应
        
        Args:
            prompt: 提示文本
            context: 上下文消息列表
            model: 模型名称，如gpt-4, gpt-3.5-turbo等
            temperature: 温度参数
            max_tokens: 最大生成令牌数
            
        Returns:
            str: 生成的文本
            
        Raises:
            LLMProviderException: OpenAI API调用异常
        """
        try:
            messages = self._prepare_messages(context, prompt)
            
            async def _generate():
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                if response.choices and len(response.choices) > 0:
                    return response.choices[0].message.content
                
                return ""
            
            return await handle_llm_exception(_generate)
        except Exception as e:
            raise self._format_exception(e)
    
    async def generate_embeddings(self, texts: List[str], model: str = None) -> List[List[float]]:
        """生成文本嵌入向量
        
        Args:
            texts: 文本列表
            model: 嵌入模型名称，默认为text-embedding-3-small
            
        Returns:
            List[List[float]]: 嵌入向量列表
            
        Raises:
            LLMProviderException: OpenAI API调用异常
        """
        try:
            if not model:
                model = "text-embedding-3-small"
            
            async def _embed():
                response = await self.client.embeddings.create(
                    model=model,
                    input=texts
                )
                
                embeddings = [data.embedding for data in response.data]
                return embeddings
            
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
                model = "gpt-3.5-turbo"
            
            # 根据模型获取编码器
            encoding = None
            try:
                # 对于的聊天模型，使用cl100k_base
                if "gpt-4" in model or "gpt-3.5-turbo" in model:
                    encoding = tiktoken.get_encoding("cl100k_base")
                else:
                    # 尝试直接使用模型名获取编码器
                    encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                # 如果找不到特定模型的编码器，使用p50k_base作为后备
                encoding = tiktoken.get_encoding("p50k_base")
            
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
            LLMProviderException: OpenAI API调用异常
        """
        try:
            messages = self._prepare_messages(context, prompt)
            
            async def _stream():
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
            
            async for chunk in await handle_llm_exception(_stream):
                yield chunk
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
                "max_tokens": 8192,  # 对于嵌入任务，无需设置max_tokens
            }
        
        # 不同模型的默认参数
        if "gpt-4" in model:
            return {
                "temperature": 0.7,
                "max_tokens": 1500,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }
        elif "gpt-3.5-turbo" in model:
            return {
                "temperature": 0.7,
                "max_tokens": 1000,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }
        else:
            return super()._get_model_defaults(model, task_type)
