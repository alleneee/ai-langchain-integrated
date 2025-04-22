"""
LangChain 通义千问(Qwen) 集成模块

这个模块提供与阿里云通义千问(Qwen)模型的LangChain集成，允许使用Qwen模型作为LangChain中的LLM。
"""

import logging
from typing import Dict, Any, List, Optional, Union
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from src.services.langchain_integrations.base import BaseLangChainIntegration

logger = logging.getLogger(__name__)

class QwenIntegration(BaseLangChainIntegration):
    """通义千问 LangChain 集成类"""
    
    @staticmethod
    def get_chat_model(config: Dict[str, Any]) -> BaseChatModel:
        """
        获取通义千问聊天模型
        
        Args:
            config: 配置字典，包含API密钥等信息
                - api_key: 通义千问 API 密钥
                - model_name: 模型名称，例如 "qwen-plus", "qwen-max" 等
                - api_base: API基础URL（可选，默认为https://dashscope.aliyuncs.com/compatible-mode/v1）
                - temperature: 温度参数（可选）
                - streaming: 是否使用流式响应（可选）
                - max_tokens: 最大生成令牌数（可选）
        
        Returns:
            BaseChatModel: 通义千问聊天模型实例
        """
        try:
            # 通义千问API兼容OpenAI格式，可以使用OpenAI的LangChain集成
            from langchain_openai import ChatOpenAI
            
            # 提取配置
            api_key = config.get("api_key", "")
            model_name = config.get("model_name", "qwen-plus")
            api_base = config.get("api_base", "https://dashscope.aliyuncs.com/compatible-mode/v1")
            temperature = config.get("temperature", 0.7)
            streaming = config.get("streaming", False)
            max_tokens = config.get("max_tokens", 2000)
            
            # 创建模型实例
            return ChatOpenAI(
                model=model_name,
                openai_api_key=api_key,
                openai_api_base=api_base,
                temperature=temperature,
                streaming=streaming,
                max_tokens=max_tokens
            )
        except ImportError:
            logger.error("未安装langchain-openai库。请使用pip install langchain-openai安装")
            raise ImportError("未安装langchain-openai库")
    
    @staticmethod
    def get_embedding_model(config: Dict[str, Any]):
        """
        获取通义千问嵌入模型
        
        Args:
            config: 配置字典
                - api_key: 通义千问 API 密钥
                - model_name: 模型名称，默认为 "text-embedding-v1"
                - api_base: API基础URL（可选，默认为https://dashscope.aliyuncs.com/compatible-mode/v1）
                
        Returns:
            OpenAIEmbeddings: 通义千问嵌入模型实例
        """
        try:
            # 通义千问嵌入API兼容OpenAI格式
            from langchain_openai import OpenAIEmbeddings
            
            # 提取配置
            api_key = config.get("api_key", "")
            model_name = config.get("model_name", "text-embedding-v1")
            api_base = config.get("api_base", "https://dashscope.aliyuncs.com/compatible-mode/v1")
            
            # 创建嵌入模型
            return OpenAIEmbeddings(
                model=model_name,
                openai_api_key=api_key,
                openai_api_base=api_base
            )
        except ImportError:
            logger.error("未安装langchain-openai库。请使用pip install langchain-openai安装")
            raise ImportError("未安装langchain-openai库")
    
    @staticmethod
    def get_default_chat_model_name() -> str:
        """
        获取默认的聊天模型名称
        
        Returns:
            str: 默认模型名称
        """
        return "qwen-plus"
    
    @staticmethod
    def get_default_embedding_model_name() -> str:
        """
        获取默认的嵌入模型名称
        
        Returns:
            str: 默认嵌入模型名称
        """
        return "text-embedding-v1"
    
    @staticmethod
    def get_available_chat_models() -> List[str]:
        """
        获取可用的聊天模型列表
        
        Returns:
            List[str]: 可用模型列表
        """
        return [
            "qwen-turbo",
            "qwen-plus",
            "qwen-max",
            "qwen-max-1201",
            "qwen-max-longcontext"
        ]
    
    @staticmethod
    def get_available_embedding_models() -> List[str]:
        """
        获取可用的嵌入模型列表
        
        Returns:
            List[str]: 可用嵌入模型列表
        """
        return [
            "text-embedding-v1",
            "text-embedding-v2"
        ]
