"""
LLM提供商适配器基类

定义了所有LLM提供商适配器必须实现的接口
"""

from abc import abstractmethod
from typing import Dict, List, Any, Optional, AsyncGenerator
from src.core.interfaces import LLMProviderInterface
from src.core.exceptions import (
    LLMProviderException, LLMProviderAuthException,
    LLMProviderQuotaException, LLMProviderRateLimitException,
    LLMProviderModelNotFoundException
)

class BaseLLMProvider(LLMProviderInterface):
    """LLM提供商基类"""
    
    def __init__(self, api_key: str = None, api_base: str = None, **kwargs):
        """初始化LLM提供商
        
        Args:
            api_key: API密钥
            api_base: API基础URL
            **kwargs: 其他参数
        """
        self.api_key = api_key
        self.api_base = api_base
        self.extra_kwargs = kwargs
        self.provider_name = self.__class__.__name__
    
    @abstractmethod
    async def generate_text(self, prompt: str, context: List[Dict], model: str, 
                         temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """生成文本响应
        
        Args:
            prompt: 提示文本
            context: 上下文消息列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大生成令牌数
            
        Returns:
            str: 生成的文本
            
        Raises:
            LLMProviderException: LLM提供商异常
        """
        pass
    
    @abstractmethod
    async def generate_embeddings(self, texts: List[str], model: str = None) -> List[List[float]]:
        """生成文本嵌入向量
        
        Args:
            texts: 文本列表
            model: 嵌入模型名称
            
        Returns:
            List[List[float]]: 嵌入向量列表
            
        Raises:
            LLMProviderException: LLM提供商异常
        """
        pass
    
    @abstractmethod
    async def count_tokens(self, text: str, model: str = None) -> Dict[str, int]:
        """计算文本的token数量
        
        Args:
            text: 待计算的文本
            model: 模型名称
            
        Returns:
            Dict[str, int]: 包含token数量和字符数量的字典
            
        Raises:
            LLMProviderException: LLM提供商异常
        """
        pass
    
    @abstractmethod
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
            LLMProviderException: LLM提供商异常
        """
        pass
    
    def _validate_api_key(self):
        """验证API密钥是否有效
        
        Raises:
            LLMProviderAuthException: API密钥无效时抛出异常
        """
        if not self.api_key:
            raise LLMProviderAuthException(f"{self.provider_name} API密钥未设置")
    
    def _get_model_defaults(self, model: str, task_type: str = "chat") -> Dict[str, Any]:
        """获取模型默认参数
        
        Args:
            model: 模型名称
            task_type: 任务类型，可选值：chat, completion, embedding
            
        Returns:
            Dict[str, Any]: 模型默认参数
        """
        # 这里子类可以覆盖以提供特定模型的默认参数
        return {
            "temperature": 0.7,
            "max_tokens": 1000 if task_type == "chat" else 16 if task_type == "embedding" else 500,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
    
    def _format_exception(self, exc: Exception) -> Exception:
        """格式化异常，将第三方SDK异常转换为应用定义的异常
        
        Args:
            exc: 原始异常
            
        Returns:
            Exception: 格式化后的异常
        """
        error_msg = str(exc)
        
        # 认证错误处理
        if any(keyword in error_msg.lower() for keyword in ["authentication", "auth", "key", "unauthorized", "权限", "认证"]):
            return LLMProviderAuthException(f"{self.provider_name} 认证失败: {error_msg}")
        
        # 配额错误处理
        if any(keyword in error_msg.lower() for keyword in ["quota", "billing", "payment", "支付", "账单", "配额"]):
            return LLMProviderQuotaException(f"{self.provider_name} 配额已用尽: {error_msg}")
        
        # 速率限制处理
        if any(keyword in error_msg.lower() for keyword in ["rate limit", "too many", "ratelimit", "速率", "频率"]):
            return LLMProviderRateLimitException(f"{self.provider_name} 请求频率过高: {error_msg}")
        
        # 模型不存在处理
        if any(keyword in error_msg.lower() for keyword in ["model not found", "model doesn't exist", "模型不存在"]):
            return LLMProviderModelNotFoundException(f"{self.provider_name} 模型不存在: {error_msg}")
        
        # 默认返回通用LLM异常
        return LLMProviderException(f"{self.provider_name} 调用失败: {error_msg}")
    
    def _prepare_messages(self, context: List[Dict], prompt: str = None) -> List[Dict[str, str]]:
        """将上下文和提示转换为标准消息格式
        
        Args:
            context: 上下文消息列表
            prompt: 当前提示
            
        Returns:
            List[Dict[str, str]]: 标准格式的消息列表
        """
        messages = []
        
        # 添加上下文消息
        if context:
            for msg in context:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role and content:
                    messages.append({"role": role, "content": content})
        
        # 添加当前提示作为用户消息
        if prompt and (not context or context[-1]["role"] != "user"):
            messages.append({"role": "user", "content": prompt})
        
        return messages
