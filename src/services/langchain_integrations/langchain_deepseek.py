"""
LangChain DeepSeek 集成模块

这个模块提供了与 langchain-deepseek 库的集成，允许使用 DeepSeek 模型作为 LangChain 中的 LLM。
"""

import logging
from typing import Dict, Any, List, Optional, Union, AsyncGenerator
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from src.services.langchain_integrations.base import BaseLangChainIntegration

logger = logging.getLogger(__name__)

class DeepSeekIntegration(BaseLangChainIntegration):
    """DeepSeek LangChain 集成类"""
    
    @staticmethod
    def get_chat_model(config: Dict[str, Any]) -> BaseChatModel:
        """
        获取 DeepSeek 聊天模型
        
        Args:
            config: 配置字典，包含 API 密钥等信息
                - api_key: DeepSeek API 密钥
                - model_name: 模型名称，例如 "deepseek-chat"
                - api_base: API 基础 URL（可选）
                - temperature: 温度参数（可选）
                - streaming: 是否使用流式响应（可选）
                - max_tokens: 最大生成令牌数（可选）
        
        Returns:
            BaseChatModel: DeepSeek 聊天模型实例
        """
        try:
            # 导入 DeepSeek LangChain 集成
            from langchain_deepseek import ChatDeepSeek
            
            # 提取配置
            api_key = config.get("api_key", "")
            api_base = config.get("api_base", None)
            model_name = config.get("model_name", "deepseek-chat")
            temperature = config.get("temperature", 0.7)
            streaming = config.get("streaming", False)
            max_tokens = config.get("max_tokens", 2000)
            
            # 创建模型实例
            return ChatDeepSeek(
                model=model_name,
                deepseek_api_key=api_key,
                deepseek_api_base=api_base,
                temperature=temperature,
                streaming=streaming,
                max_tokens=max_tokens
            )
        except ImportError:
            logger.error("未安装 langchain-deepseek 库。请使用 pip install langchain-deepseek==0.1.3 安装")
            raise ImportError("未安装 langchain-deepseek 库")
    
    @staticmethod
    def get_embedding_model(config: Dict[str, Any]):
        """
        获取 DeepSeek 嵌入模型
        
        Args:
            config: 配置字典
                - api_key: DeepSeek API 密钥
                - model_name: 模型名称，默认为 "deepseek-embedding"
                - api_base: API 基础 URL（可选）
                
        Returns:
            DeepSeekEmbeddings: DeepSeek 嵌入模型实例
        """
        try:
            # 导入 DeepSeek Embeddings
            from langchain_deepseek import DeepSeekEmbeddings
            
            # 提取配置
            api_key = config.get("api_key", "")
            api_base = config.get("api_base", None)
            model_name = config.get("model_name", "deepseek-embedding")
            
            # 创建嵌入模型
            return DeepSeekEmbeddings(
                model=model_name,
                deepseek_api_key=api_key,
                deepseek_api_base=api_base
            )
        except ImportError:
            logger.error("未安装 langchain-deepseek 库。请使用 pip install langchain-deepseek==0.1.3 安装")
            raise ImportError("未安装 langchain-deepseek 库")
    
    @staticmethod
    def get_default_chat_model_name() -> str:
        """
        获取默认的聊天模型名称
        
        Returns:
            str: 默认模型名称
        """
        return "deepseek-chat"
    
    @staticmethod
    def get_default_embedding_model_name() -> str:
        """
        获取默认的嵌入模型名称
        
        Returns:
            str: 默认嵌入模型名称
        """
        return "deepseek-embedding"
    
    @staticmethod
    def get_available_chat_models() -> List[str]:
        """
        获取可用的聊天模型列表
        
        Returns:
            List[str]: 可用模型列表
        """
        return [
            "deepseek-chat",
            "deepseek-coder"
        ]
    
    @staticmethod
    def get_available_embedding_models() -> List[str]:
        """
        获取可用的嵌入模型列表
        
        Returns:
            List[str]: 可用嵌入模型列表
        """
        return [
            "deepseek-embedding"
        ]
