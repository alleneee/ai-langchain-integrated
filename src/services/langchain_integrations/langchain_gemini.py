"""
LangChain Google Gemini 集成模块

这个模块提供了与 langchain-google-community 库的集成，允许使用 Google Gemini 模型作为 LangChain 中的 LLM。
"""

import logging
from typing import Dict, Any, List, Optional, Union
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from src.services.langchain_integrations.base import BaseLangChainIntegration

logger = logging.getLogger(__name__)

class GeminiIntegration(BaseLangChainIntegration):
    """Google Gemini LangChain 集成类"""
    
    @staticmethod
    def get_chat_model(config: Dict[str, Any]) -> BaseChatModel:
        """
        获取 Google Gemini 聊天模型
        
        Args:
            config: 配置字典，包含 API 密钥等信息
                - api_key: Google API 密钥
                - model_name: 模型名称，例如 "gemini-pro" 或 "gemini-ultra"
                - temperature: 温度参数（可选）
                - streaming: 是否使用流式响应（可选）
                - max_tokens: 最大生成令牌数（可选）
        
        Returns:
            BaseChatModel: Gemini 聊天模型实例
        """
        try:
            # 导入 Google Gemini LangChain 集成
            from langchain_google_community import ChatGoogleGenerativeAI
            
            # 提取配置
            api_key = config.get("api_key", "")
            model_name = config.get("model_name", "gemini-pro")
            temperature = config.get("temperature", 0.7)
            streaming = config.get("streaming", False)
            max_tokens = config.get("max_tokens", 2048)
            
            # 创建模型实例
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=temperature,
                streaming=streaming,
                max_output_tokens=max_tokens,
                convert_system_message_to_human=True  # Gemini不直接支持系统消息，需要将其转换为人类消息
            )
        except ImportError:
            logger.error("未安装 langchain-google-community 库。请使用 pip install langchain-google-community 安装")
            raise ImportError("未安装 langchain-google-community 库")
    
    @staticmethod
    def get_embedding_model(config: Dict[str, Any]):
        """
        获取 Google 嵌入模型
        
        Args:
            config: 配置字典
                - api_key: Google API 密钥
                - model_name: 模型名称，默认为 "embedding-001"
                
        Returns:
            GoogleGenerativeAIEmbeddings: Google 嵌入模型实例
        """
        try:
            # 导入 Google Embeddings
            from langchain_google_community import GoogleGenerativeAIEmbeddings
            
            # 提取配置
            api_key = config.get("api_key", "")
            model_name = config.get("model_name", "embedding-001")
            
            # 创建嵌入模型
            return GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=api_key
            )
        except ImportError:
            logger.error("未安装 langchain-google-community 库。请使用 pip install langchain-google-community 安装")
            raise ImportError("未安装 langchain-google-community 库")
    
    @staticmethod
    def get_default_chat_model_name() -> str:
        """
        获取默认的聊天模型名称
        
        Returns:
            str: 默认模型名称
        """
        return "gemini-pro"
    
    @staticmethod
    def get_default_embedding_model_name() -> str:
        """
        获取默认的嵌入模型名称
        
        Returns:
            str: 默认嵌入模型名称
        """
        return "embedding-001"
    
    @staticmethod
    def get_available_chat_models() -> List[str]:
        """
        获取可用的聊天模型列表
        
        Returns:
            List[str]: 可用模型列表
        """
        return [
            "gemini-pro",
            "gemini-ultra",
            "gemini-pro-vision",
            "gemini-ultra-vision"
        ]
    
    @staticmethod
    def get_available_embedding_models() -> List[str]:
        """
        获取可用的嵌入模型列表
        
        Returns:
            List[str]: 可用嵌入模型列表
        """
        return [
            "embedding-001"
        ]
