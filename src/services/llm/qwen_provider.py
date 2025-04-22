"""
通义千问(Qwen) LLM提供商适配器

实现了阿里云通义千问 API的调用与适配
"""

import json
import tiktoken
from typing import Dict, List, Any, Optional, AsyncGenerator
from src.services.llm.base import BaseLLMProvider
from src.core.exceptions import (
    LLMProviderException, LLMProviderAuthException,
    LLMProviderModelNotFoundException
)
import dashscope
from dashscope import Generation
import asyncio

class QwenProvider(BaseLLMProvider):
    """通义千问 API适配器"""
    
    def __init__(self, api_key: str = None, api_base: str = None, **kwargs):
        """初始化通义千问适配器
        
        Args:
            api_key: 通义千问 API密钥
            api_base: 自定义API基础URL (通常不需要)
            **kwargs: 其他参数
        """
        super().__init__(api_key, api_base, **kwargs)
        self._setup_client()
    
    def _setup_client(self):
        """设置通义千问客户端"""
        try:
            self._validate_api_key()
            # 设置DashScope API密钥
            dashscope.api_key = self.api_key
            # 不需要创建客户端实例，dashscope使用全局API密钥
        except Exception as e:
            raise self._format_exception(e)
    
    async def generate_text(self, prompt: str, context: List[Dict], model: str,
                         temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """生成文本响应
        
        Args:
            prompt: 提示文本
            context: 上下文消息列表
            model: 模型名称，如qwen-turbo、qwen-plus等
            temperature: 温度参数
            max_tokens: 最大生成令牌数
            
        Returns:
            str: 生成的文本
            
        Raises:
            LLMProviderException: 通义千问 API调用异常
        """
        try:
            messages = self._prepare_messages(context, prompt)
            
            # 转换为dashscope格式的消息
            dashscope_messages = []
            for msg in messages:
                dashscope_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # DashScope API是同步的，使用run_in_executor运行
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: Generation.call(
                    model=model,
                    messages=dashscope_messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            )
            
            # 检查响应
            if response.status_code == 200:
                if response.output and response.output.choices and len(response.output.choices) > 0:
                    return response.output.choices[0].message.content
            
            # 如果响应不成功，抛出异常
            if response.status_code != 200:
                raise Exception(f"通义千问API调用失败: {response.code} - {response.message}")
            
            return ""
        except Exception as e:
            raise self._format_exception(e)
    
    async def generate_embeddings(self, texts: List[str], model: str = None) -> List[List[float]]:
        """生成文本嵌入向量
        
        Args:
            texts: 文本列表
            model: 嵌入模型名称，默认为text-embedding-v1
            
        Returns:
            List[List[float]]: 嵌入向量列表
            
        Raises:
            LLMProviderException: 通义千问 API调用异常
        """
        try:
            from dashscope import TextEmbedding
            
            if not model:
                model = "text-embedding-v1"
            
            # DashScope API是同步的，使用run_in_executor运行
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: TextEmbedding.call(
                    model=model,
                    input=texts
                )
            )
            
            # 检查响应
            if response.status_code == 200 and response.output and response.output.embeddings:
                embeddings = [data.embedding for data in response.output.embeddings]
                return embeddings
            
            # 如果响应不成功，抛出异常
            if response.status_code != 200:
                raise Exception(f"通义千问嵌入API调用失败: {response.code} - {response.message}")
            
            return []
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
            from dashscope import TextEmbedding
            
            # 对于中文英文混合文本，使用cl100k_base
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
            LLMProviderException: 通义千问 API调用异常
        """
        try:
            messages = self._prepare_messages(context, prompt)
            
            # 转换为dashscope格式的消息
            dashscope_messages = []
            for msg in messages:
                dashscope_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # 使用异步生成器包装同步流式响应
            async def stream_response():
                # DashScope API是同步的，使用run_in_executor运行
                loop = asyncio.get_event_loop()
                
                def get_stream_response():
                    # 使用stream=True启用流式响应
                    return Generation.call(
                        model=model,
                        messages=dashscope_messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        result_format='message',
                        stream=True
                    )
                
                # 由于DashScope流式API返回一个迭代器，我们需要手动迭代它
                response_iter = await loop.run_in_executor(None, get_stream_response)
                
                # 手动迭代同步响应
                for event in response_iter:
                    if event.status_code == 200:
                        if event.output and event.output.choices and len(event.output.choices) > 0:
                            content = event.output.choices[0].message.content
                            if content:
                                yield content
                    else:
                        # 如果发生错误，打印错误信息并停止
                        raise Exception(f"通义千问流式API调用失败: {event.code} - {event.message}")
            
            # 返回异步生成器
            async for chunk in stream_response():
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
                "max_tokens": 8192,
            }
        
        # 通义千问模型默认参数
        if "qwen-turbo" in model:
            return {
                "temperature": 0.7,
                "max_tokens": 1500,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }
        elif "qwen-plus" in model:
            return {
                "temperature": 0.7,
                "max_tokens": 2000,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }
        elif "qwen-max" in model:
            return {
                "temperature": 0.7,
                "max_tokens": 4000,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }
        else:
            return super()._get_model_defaults(model, task_type)
