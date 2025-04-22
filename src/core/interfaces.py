"""
接口定义模块

该模块定义了系统中的核心接口和抽象类，用于实现依赖注入和解耦
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union

class LLMProviderInterface(ABC):
    """LLM服务提供商接口"""
    
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
                       temperature: float = 0.7, max_tokens: int = 1000):
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
    
class FactoryInterface(ABC):
    """工厂接口基类"""
    
    @abstractmethod
    async def create(self, *args, **kwargs) -> Any:
        """创建对象
        
        Returns:
            Any: 创建的对象
            
        Raises:
            FactoryException: 工厂创建对象异常
        """
        pass

class ServiceInterface(ABC):
    """服务接口基类"""
    
    @abstractmethod
    async def initialize(self) -> None:
        """初始化服务
        
        Raises:
            ServiceInitException: 服务初始化异常
        """
        pass
