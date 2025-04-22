"""
LLM服务模块

该模块提供了LLM相关的服务实现。
"""

import logging
from typing import Dict, List, Any, Optional
from src.services.base import BaseService
from src.core.interfaces import LLMProviderInterface
from src.factories.llm_provider_factory import LLMProviderFactory
from src.core.exceptions import (
    LLMProviderException, ConfigurationException,
    ServiceInitException, FactoryException
)

logger = logging.getLogger(__name__)

class LLMService(BaseService):
    """LLM服务实现"""
    
    async def initialize(self):
        """初始化LLM服务
        
        Raises:
            ServiceInitException: 服务初始化异常
        """
        try:
            # 初始化提供商工厂
            self.provider_factory = LLMProviderFactory()
            
            # 预加载默认提供商
            from src.config.settings import settings
            default_provider = settings.DEFAULT_LLM_PROVIDER
            if default_provider:
                try:
                    await self.provider_factory.create(default_provider)
                    logger.info(f"默认LLM提供商 {default_provider} 已初始化")
                except Exception as e:
                    logger.warning(f"默认LLM提供商 {default_provider} 初始化失败: {str(e)}")
        except Exception as e:
            logger.error(f"LLM服务初始化失败: {str(e)}")
            raise ServiceInitException(f"LLM服务初始化失败: {str(e)}")
    
    async def get_llm_provider(self, provider_name: str, **kwargs) -> LLMProviderInterface:
        """获取LLM提供商实例
        
        Args:
            provider_name: 提供商名称
            **kwargs: 覆盖默认配置的参数
            
        Returns:
            LLMProviderInterface: LLM提供商实例
            
        Raises:
            LLMProviderException: 获取提供商实例失败
        """
        try:
            return await self.provider_factory.create(provider_name, **kwargs)
        except FactoryException as e:
            logger.error(f"获取LLM提供商实例失败: {str(e)}")
            raise LLMProviderException(f"获取提供商实例失败: {str(e)}")
    
    async def generate_text(self, prompt: str, provider: str = None, model: str = None, 
                      context: List[Dict] = None, **kwargs) -> str:
        """生成文本
        
        Args:
            prompt: 提示文本
            provider: 提供商名称
            model: 模型名称
            context: 上下文信息
            **kwargs: 其他参数
            
        Returns:
            str: 生成的文本
            
        Raises:
            LLMProviderException: 生成文本失败
        """
        from src.config.settings import settings
        
        # 使用默认提供商和模型（如果未指定）
        provider = provider or settings.DEFAULT_LLM_PROVIDER
        model = model or settings.DEFAULT_LLM_MODEL
        context = context or []
        
        try:
            # 获取提供商实例
            llm_provider = await self.get_llm_provider(provider)
            
            # 生成文本
            result = await llm_provider.generate_text(
                prompt=prompt,
                context=context,
                model=model,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 1000)
            )
            
            return result
        except Exception as e:
            logger.error(f"生成文本失败: {str(e)}")
            if isinstance(e, LLMProviderException):
                raise e
            else:
                raise LLMProviderException(f"生成文本失败: {str(e)}")
    
    async def generate_embeddings(self, texts: List[str], provider: str = None, 
                             model: str = None) -> List[List[float]]:
        """生成文本嵌入向量
        
        Args:
            texts: 文本列表
            provider: 提供商名称
            model: 嵌入模型名称
            
        Returns:
            List[List[float]]: 嵌入向量列表
            
        Raises:
            LLMProviderException: 生成嵌入向量失败
        """
        from src.config.settings import settings
        
        # 使用默认提供商和嵌入模型（如果未指定）
        provider = provider or settings.DEFAULT_EMBEDDING_PROVIDER
        model = model or settings.OPENAI_EMBEDDING_MODEL
        
        try:
            # 获取提供商实例
            llm_provider = await self.get_llm_provider(provider)
            
            # 生成嵌入向量
            embeddings = await llm_provider.generate_embeddings(
                texts=texts,
                model=model
            )
            
            return embeddings
        except Exception as e:
            logger.error(f"生成嵌入向量失败: {str(e)}")
            if isinstance(e, LLMProviderException):
                raise e
            else:
                raise LLMProviderException(f"生成嵌入向量失败: {str(e)}")
    
    async def estimate_token_count(self, text: str, model: str = None) -> Dict[str, int]:
        """估计文本的token数量
        
        Args:
            text: 待计算的文本
            model: 模型名称
            
        Returns:
            Dict[str, int]: 包含token数量和字符数量的字典
            
        Raises:
            LLMProviderException: 计算token失败
        """
        from src.config.settings import settings
        
        # 使用默认提供商和模型（如果未指定）
        provider = settings.DEFAULT_LLM_PROVIDER
        model = model or settings.DEFAULT_LLM_MODEL
        
        try:
            # 获取提供商实例
            llm_provider = await self.get_llm_provider(provider)
            
            # 计算token
            result = await llm_provider.count_tokens(
                text=text,
                model=model
            )
            
            return result
        except Exception as e:
            logger.error(f"估计token数量失败: {str(e)}")
            if isinstance(e, LLMProviderException):
                raise e
            else:
                raise LLMProviderException(f"估计token数量失败: {str(e)}")
    
    async def get_supported_providers(self) -> List[Dict[str, Any]]:
        """获取支持的LLM提供商列表
        
        Returns:
            List[Dict[str, Any]]: 支持的提供商列表
            
        Raises:
            LLMProviderException: 获取提供商列表失败
        """
        try:
            providers = await self.provider_factory.get_supported_providers()
            return providers
        except Exception as e:
            logger.error(f"获取支持的提供商列表失败: {str(e)}")
            raise LLMProviderException(f"获取支持的提供商列表失败: {str(e)}")
    
    async def stream_chat(self, prompt: str, provider: str = None, model: str = None,
                     context: List[Dict] = None, **kwargs):
        """流式生成聊天回复
        
        Args:
            prompt: 提示文本
            provider: 提供商名称
            model: 模型名称
            context: 上下文信息
            **kwargs: 其他参数
            
        Yields:
            str: 生成的文本片段
            
        Raises:
            LLMProviderException: 流式生成失败
        """
        from src.config.settings import settings
        
        # 使用默认提供商和模型（如果未指定）
        provider = provider or settings.DEFAULT_LLM_PROVIDER
        model = model or settings.DEFAULT_LLM_MODEL
        context = context or []
        
        try:
            # 获取提供商实例
            llm_provider = await self.get_llm_provider(provider)
            
            # 流式生成文本
            async for chunk in llm_provider.stream_chat(
                prompt=prompt,
                context=context,
                model=model,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 1000)
            ):
                yield chunk
        except Exception as e:
            logger.error(f"流式生成聊天回复失败: {str(e)}")
            if isinstance(e, LLMProviderException):
                raise e
            else:
                raise LLMProviderException(f"流式生成聊天回复失败: {str(e)}")