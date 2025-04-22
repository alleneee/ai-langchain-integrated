"""
LangChain 集成基类模块

这个模块提供了所有 LangChain 集成的基类
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class BaseLangChainIntegration(ABC):
    """LangChain 集成基类"""
    
    @staticmethod
    @abstractmethod
    def get_chat_model(config: Dict[str, Any]):
        """
        获取聊天模型
        
        Args:
            config: 配置字典
            
        Returns:
            聊天模型实例
        """
        pass
    
    @staticmethod
    @abstractmethod
    def get_embedding_model(config: Dict[str, Any]):
        """
        获取嵌入模型
        
        Args:
            config: 配置字典
            
        Returns:
            嵌入模型实例
        """
        pass
    
    @staticmethod
    @abstractmethod
    def get_default_chat_model_name() -> str:
        """
        获取默认的聊天模型名称
        
        Returns:
            str: 默认模型名称
        """
        pass
    
    @staticmethod
    @abstractmethod
    def get_default_embedding_model_name() -> str:
        """
        获取默认的嵌入模型名称
        
        Returns:
            str: 默认嵌入模型名称
        """
        pass
    
    @staticmethod
    @abstractmethod
    def get_available_chat_models() -> List[str]:
        """
        获取可用的聊天模型列表
        
        Returns:
            List[str]: 可用模型列表
        """
        pass
    
    @staticmethod
    @abstractmethod
    def get_available_embedding_models() -> List[str]:
        """
        获取可用的嵌入模型列表
        
        Returns:
            List[str]: 可用嵌入模型列表
        """
        pass
