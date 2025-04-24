"""
LLM服务模块

该模块提供了LLM相关的服务实现，包含异步性能优化和连接池管理。
"""

import logging
import asyncio
import time
from typing import Dict, List, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from src.services.base import BaseService
from src.core.interfaces import LLMProviderInterface
from src.factories.llm_provider_factory import LLMProviderFactory
from src.core.exceptions import (
    LLMProviderException, ConfigurationException,
    ServiceInitException, FactoryException
)
from src.core.connection_pool import ConnectionPoolManager

logger = logging.getLogger(__name__)

class LLMService(BaseService):
    """LLM服务实现，带异步优化"""
    
    async def initialize(self):
        """初始化LLM服务
        
        Raises:
            ServiceInitException: 服务初始化异常
        """
        try:
            # 初始化提供商工厂
            self.provider_factory = LLMProviderFactory()
            
            # 添加请求速率限制器（信号量和令牌桶）
            self.request_semaphore = asyncio.Semaphore(50)  # 最多50个并发LLM请求
            self.embedding_semaphore = asyncio.Semaphore(100)  # 最多100个并发嵌入请求
            
            # 添加缓存
            self.token_count_cache = {}  # 简单的token计数缓存
            self.embedding_cache = {}  # 简单的嵌入缓存
            
            # 获取连接池管理器
            self.pool_manager = ConnectionPoolManager.get_instance()
            
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
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    async def get_llm_provider(self, provider_name: str, **kwargs) -> LLMProviderInterface:
        """获取LLM提供商实例，带重试机制
        
        Args:
            provider_name: 提供商名称
            **kwargs: 覆盖默认配置的参数
            
        Returns:
            LLMProviderInterface: LLM提供商实例
            
        Raises:
            LLMProviderException: 获取提供商实例失败
        """
        try:
            # 添加HTTP会话到kwargs
            if 'http_session' not in kwargs:
                kwargs['http_session'] = self.pool_manager.get_http_session("llm")
                
            return await self.provider_factory.create(provider_name, **kwargs)
        except FactoryException as e:
            logger.error(f"获取LLM提供商实例失败: {str(e)}")
            raise LLMProviderException(f"获取提供商实例失败: {str(e)}")
    
    async def generate_text(self, prompt: str, provider: str = None, model: str = None, 
                      context: List[Dict] = None, **kwargs) -> str:
        """生成文本，带并发控制
        
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
        
        # 使用信号量限制并发请求
        async with self.request_semaphore:
            try:
                # 记录开始时间
                start_time = time.time()
                
                # 获取提供商实例
                llm_provider = await self.get_llm_provider(provider)
                
                # 增加超时控制
                timeout = kwargs.pop("timeout", 60)  # 默认60秒超时
                
                # 生成文本，带超时
                result = await asyncio.wait_for(
                    llm_provider.generate_text(
                        prompt=prompt,
                        context=context,
                        model=model,
                        temperature=kwargs.get("temperature", 0.7),
                        max_tokens=kwargs.get("max_tokens", 1000),
                        **kwargs
                    ),
                    timeout=timeout
                )
                
                # 记录耗时
                elapsed = time.time() - start_time
                logger.info(f"LLM请求耗时: {elapsed:.2f}秒, 模型: {model}, 提供商: {provider}")
                
                return result
            except asyncio.TimeoutError:
                logger.error(f"LLM请求超时: prompt={prompt[:50]}..., model={model}, provider={provider}")
                raise LLMProviderException("LLM请求超时")
            except Exception as e:
                logger.error(f"生成文本失败: {str(e)}")
                if isinstance(e, LLMProviderException):
                    raise e
                else:
                    raise LLMProviderException(f"生成文本失败: {str(e)}")
    
    async def batch_generate_embeddings(self, texts_batches: List[List[str]], provider: str = None, 
                                  model: str = None) -> List[List[List[float]]]:
        """批量生成多组文本的嵌入向量
        
        Args:
            texts_batches: 多组文本列表
            provider: 提供商名称
            model: 嵌入模型名称
            
        Returns:
            List[List[List[float]]]: 多组嵌入向量列表
            
        Raises:
            LLMProviderException: 生成嵌入向量失败
        """
        # 并行处理多批次嵌入
        tasks = []
        for texts in texts_batches:
            tasks.append(self.generate_embeddings(texts, provider, model))
        
        return await asyncio.gather(*tasks)
    
    async def generate_embeddings(self, texts: List[str], provider: str = None, 
                             model: str = None) -> List[List[float]]:
        """生成文本嵌入向量，带并发控制和缓存
        
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
        
        # 使用信号量限制并发请求
        async with self.embedding_semaphore:
            try:
                # 查找缓存
                cache_key = f"{provider}:{model}"
                uncached_texts = []
                uncached_indices = []
                result_embeddings = [None] * len(texts)
                
                # 检查哪些文本需要生成嵌入
                for i, text in enumerate(texts):
                    text_key = f"{cache_key}:{hash(text)}"
                    if text_key in self.embedding_cache:
                        result_embeddings[i] = self.embedding_cache[text_key]
                    else:
                        uncached_texts.append(text)
                        uncached_indices.append(i)
                
                # 如果所有文本都已缓存，直接返回
                if not uncached_texts:
                    return result_embeddings
                
                # 获取提供商实例
                llm_provider = await self.get_llm_provider(provider)
                
                # 生成未缓存文本的嵌入向量
                embeddings = await asyncio.wait_for(
                    llm_provider.generate_embeddings(
                        texts=uncached_texts,
                        model=model
                    ),
                    timeout=30  # 30秒超时
                )
                
                # 更新缓存和结果
                for i, embedding in zip(uncached_indices, embeddings):
                    text = texts[i]
                    text_key = f"{cache_key}:{hash(text)}"
                    self.embedding_cache[text_key] = embedding
                    result_embeddings[i] = embedding
                
                return result_embeddings
            except asyncio.TimeoutError:
                logger.error(f"嵌入请求超时: texts={len(texts)}条, model={model}, provider={provider}")
                raise LLMProviderException("嵌入请求超时")
            except Exception as e:
                logger.error(f"生成嵌入向量失败: {str(e)}")
                if isinstance(e, LLMProviderException):
                    raise e
                else:
                    raise LLMProviderException(f"生成嵌入向量失败: {str(e)}")
    
    async def estimate_token_count(self, text: str, model: str = None) -> Dict[str, int]:
        """估计文本的token数量，带缓存
        
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
        
        # 检查缓存
        cache_key = f"{model}:{hash(text)}"
        if cache_key in self.token_count_cache:
            return self.token_count_cache[cache_key]
        
        try:
            # 获取提供商实例
            llm_provider = await self.get_llm_provider(provider)
            
            # 计算token
            result = await llm_provider.count_tokens(
                text=text,
                model=model
            )
            
            # 更新缓存
            self.token_count_cache[cache_key] = result
            
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
        """流式生成聊天回复，带并发控制
        
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
        
        # 使用信号量限制并发请求
        async with self.request_semaphore:
            try:
                # 记录开始时间
                start_time = time.time()
                
                # 获取提供商实例
                llm_provider = await self.get_llm_provider(provider)
                
                # 流式生成文本
                chunk_count = 0
                async for chunk in llm_provider.stream_chat(
                    prompt=prompt,
                    context=context,
                    model=model,
                    temperature=kwargs.get("temperature", 0.7),
                    max_tokens=kwargs.get("max_tokens", 1000),
                    **kwargs
                ):
                    chunk_count += 1
                    yield chunk
                
                # 记录耗时
                elapsed = time.time() - start_time
                logger.info(f"流式LLM请求耗时: {elapsed:.2f}秒, 收到{chunk_count}个块, 模型: {model}, 提供商: {provider}")
                
            except Exception as e:
                logger.error(f"流式生成聊天回复失败: {str(e)}")
                if isinstance(e, LLMProviderException):
                    raise e
                else:
                    raise LLMProviderException(f"流式生成聊天回复失败: {str(e)}")
    
    # 新增方法：清理缓存
    async def clear_cache(self):
        """清理缓存数据"""
        self.token_count_cache.clear()
        self.embedding_cache.clear()
        logger.info("LLM服务缓存已清理")
        
    # 新增方法：并行调用多个LLM
    async def parallel_generate_text(self, 
                              prompts: List[str], 
                              providers: List[str] = None,
                              models: List[str] = None,
                              **kwargs) -> List[str]:
        """并行调用多个LLM生成文本
        
        Args:
            prompts: 提示文本列表
            providers: 提供商名称列表，长度应与prompts相同或为None
            models: 模型名称列表，长度应与prompts相同或为None
            **kwargs: 其他参数
            
        Returns:
            List[str]: 生成的文本列表
            
        Raises:
            LLMProviderException: 生成文本失败
        """
        from src.config.settings import settings
        
        # 准备参数
        count = len(prompts)
        if providers and len(providers) != count:
            raise ValueError("providers列表长度必须与prompts相同")
        if models and len(models) != count:
            raise ValueError("models列表长度必须与prompts相同")
        
        default_provider = settings.DEFAULT_LLM_PROVIDER
        default_model = settings.DEFAULT_LLM_MODEL
        
        providers = providers or [default_provider] * count
        models = models or [default_model] * count
        
        # 创建并行任务
        tasks = []
        for i in range(count):
            tasks.append(self.generate_text(
                prompt=prompts[i],
                provider=providers[i],
                model=models[i],
                **kwargs
            ))
        
        # 并行执行所有任务
        return await asyncio.gather(*tasks)