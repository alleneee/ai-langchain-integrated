"""
Chatbot服务模块

该模块提供了Chatbot服务类，集成了LangChain，用于处理聊天请求和响应。
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple

from src.services.base import BaseService
from src.services.langchain_service import LangChainService
from src.config.settings import settings
from src.factories import LLMFactory

logger = logging.getLogger(__name__)

class ChatbotService(BaseService):
    """Chatbot服务类，提供统一的聊天接口"""
    
    async def initialize(self):
        """初始化服务"""
        self._models_cache = {}  # 缓存已创建的模型实例
        self._langchain_service = LangChainService()
        await self._langchain_service.initialize()
    
    async def get_supported_providers(self) -> List[str]:
        """
        获取支持的LLM提供商列表
        
        Returns:
            支持的提供商列表
        """
        return LLMFactory.get_supported_providers()
    
    def _create_config(self, provider: str, **kwargs) -> Dict[str, Any]:
        """
        根据提供商创建配置
        
        Args:
            provider: LLM提供商名称
            **kwargs: 配置参数
            
        Returns:
            配置字典
            
        Raises:
            ValueError: 如果提供商不支持
        """
        provider = provider.lower()
        
        # 默认配置参数
        model_name = kwargs.get('model_name')
        temperature = kwargs.get('temperature', 0.7)
        max_tokens = kwargs.get('max_tokens')
        streaming = kwargs.get('streaming', False)
        api_key = kwargs.get('api_key')
        top_p = kwargs.get('top_p', 1.0)
        
        config = {
            "model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "streaming": streaming,
            "api_key": api_key,
            "top_p": top_p
        }
        
        # 根据提供商添加特定参数
        if provider == 'openai':
            if not model_name:
                config["model_name"] = "gpt-3.5-turbo"
            config["base_url"] = kwargs.get('api_base', settings.OPENAI_API_BASE)
            config["api_version"] = kwargs.get('api_version')
            config["organization"] = kwargs.get('organization_id')
            if not api_key and hasattr(settings, 'OPENAI_API_KEY'):
                config["api_key"] = settings.OPENAI_API_KEY
            
        elif provider == 'anthropic' or provider == 'claude':
            if not model_name:
                config["model_name"] = "claude-3-sonnet-20240229"
            config["base_url"] = kwargs.get('api_base')
            if not api_key and hasattr(settings, 'ANTHROPIC_API_KEY'):
                config["api_key"] = settings.ANTHROPIC_API_KEY
            
        elif provider == 'azure_openai':
            if not model_name:
                config["model_name"] = "gpt-4"
            config["base_url"] = kwargs.get('api_base')
            config["api_version"] = kwargs.get('api_version', "2023-07-01-preview")
            config["deployment_name"] = kwargs.get('deployment_name')
            if not api_key and hasattr(settings, 'AZURE_OPENAI_API_KEY'):
                config["api_key"] = settings.AZURE_OPENAI_API_KEY
            
        elif provider == 'qwen':
            if not model_name:
                config["model_name"] = "qwen-turbo"
            config["base_url"] = kwargs.get('api_base', settings.QWEN_API_BASE if hasattr(settings, 'QWEN_API_BASE') else None)
            config["model_kwargs"] = kwargs.get('model_kwargs', {})
            if not api_key and hasattr(settings, 'QWEN_API_KEY'):
                config["api_key"] = settings.QWEN_API_KEY
            
        elif provider == 'gemini' or provider == 'google_ai':
            if not model_name:
                config["model_name"] = "gemini-pro"
            if not api_key and hasattr(settings, 'GOOGLE_API_KEY'):
                config["api_key"] = settings.GOOGLE_API_KEY
            
        elif provider == 'ollama':
            if not model_name:
                config["model_name"] = "llama3"
            config["base_url"] = kwargs.get('api_base', "http://localhost:11434")
            
        else:
            supported_providers = LLMFactory.get_supported_providers()
            raise ValueError(f"不支持的提供商: {provider}，支持的提供商包括: {', '.join(supported_providers)}")
            
        return config
    
    def _get_model_cache_key(self, provider: str, config: Dict[str, Any]) -> str:
        """
        获取模型缓存的键
        
        Args:
            provider: 提供商名称
            config: 配置字典
            
        Returns:
            缓存键
        """
        model_name = config.get("model_name", "default")
        api_base = config.get("base_url", "default")
        return f"{provider}_{model_name}_{api_base}"
    
    async def _get_or_create_model(self, provider: str, config: Dict[str, Any]) -> Any:
        """
        获取或创建模型实例
        
        Args:
            provider: 提供商名称
            config: 配置字典
            
        Returns:
            模型实例
        """
        cache_key = self._get_model_cache_key(provider, config)
        
        # 检查缓存
        if cache_key in self._models_cache:
            return self._models_cache[cache_key]
        
        # 创建新模型
        model = self._langchain_service.create_chat_model(provider, config)
        
        # 存入缓存
        self._models_cache[cache_key] = model
        
        return model
    
    async def chat(
        self,
        provider: str,
        messages: List[Dict[str, Any]],
        system_message: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行聊天请求
        
        Args:
            provider: LLM提供商名称
            messages: 聊天消息历史
            system_message: 系统消息
            **kwargs: 额外的配置参数
            
        Returns:
            包含聊天回复和相关信息的字典
            
        Raises:
            ValueError: 如果提供商不支持或配置错误
        """
        try:
            # 创建配置
            config = self._create_config(provider, **kwargs)
            
            # 获取或创建模型
            model = await self._get_or_create_model(provider, config)
            
            # 处理聊天请求
            content, token_usage = self._langchain_service.process_chat(
                model=model,
                messages=messages,
                system_message=system_message
            )
            
            # 返回结果
            return {
                "content": content,
                "token_usage": token_usage,
                "model": config["model_name"],
                "provider": provider
            }
            
        except Exception as e:
            logger.error(f"聊天请求失败: {str(e)}")
            raise
    
    async def chat_with_rag(
        self,
        provider: str,
        messages: List[Dict[str, Any]],
        retriever: Any,
        system_message: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行基于检索增强的聊天请求
        
        Args:
            provider: LLM提供商名称
            messages: 聊天消息历史
            retriever: 检索器实例
            system_message: 系统消息
            **kwargs: 额外的配置参数
            
        Returns:
            包含聊天回复和相关信息的字典
        """
        try:
            # 创建配置
            config = self._create_config(provider, **kwargs)
            
            # 获取或创建模型
            model = await self._get_or_create_model(provider, config)
            
            # 提取最后一条用户消息作为查询
            user_query = ""
            for message in reversed(messages):
                if message.get("role") == "user":
                    user_query = message.get("content", "")
                    break
                    
            if not user_query:
                raise ValueError("没有找到用户查询消息")
                
            # 创建检索链
            chain = self._langchain_service.create_retrieval_chain(
                llm=model,
                retriever=retriever,
                system_message=system_message
            )
            
            # 执行检索链
            response = chain.invoke(user_query)
            
            # 估算token使用情况（这只是一个估计值）
            input_tokens = len(user_query.split())
            output_tokens = len(response.split())
            
            # 返回结果
            return {
                "content": response,
                "token_usage": {
                    "input_tokens": input_tokens * 2,  # 粗略估计
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens * 2 + output_tokens
                },
                "model": config["model_name"],
                "provider": provider
            }
            
        except Exception as e:
            logger.error(f"RAG聊天请求失败: {str(e)}")
            raise
    
    def get_default_system_message(self) -> str:
        """
        获取默认的系统消息
        
        Returns:
            默认系统消息
        """
        return "你是一个由AI训练的助手。你会提供有帮助、安全、准确的回答。" 