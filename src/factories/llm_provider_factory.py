"""
LLM提供商工厂模块

该模块负责创建和管理不同的LLM提供商实例。
"""

import importlib
import logging
from typing import Dict, Any, Type, Optional, List
from src.core.interfaces import FactoryInterface
from src.core.exceptions import FactoryException, ConfigurationException
from src.services.llm.base import BaseLLMProvider
from src.config.settings import settings

logger = logging.getLogger(__name__)

class LLMProviderFactory(FactoryInterface):
    """LLM提供商工厂类"""
    
    # 提供商类映射
    PROVIDER_MAPPING = {
        "openai": "src.services.llm.openai_provider.OpenAIProvider",
        "azure_openai": "src.services.llm.azure_openai_provider.AzureOpenAIProvider",
        "anthropic": "src.services.llm.anthropic_provider.AnthropicProvider",
        "gemini": "src.services.llm.gemini_provider.GeminiProvider",
        "ollama": "src.services.llm.ollama_provider.OllamaProvider",
        "deepseek": "src.services.llm.deepseek_provider.DeepseekProvider",
        "qwen": "src.services.llm.qwen_provider.QwenProvider",
        "xunfei": "src.services.llm.xunfei_provider.XunfeiProvider",
        "xai": "src.services.llm.xai_provider.XAIProvider",
    }
    
    # 默认模型映射
    DEFAULT_MODEL_MAPPING = {
        "openai": settings.llm.openai_default_model,
        "azure_openai": settings.llm.openai_default_model,
        "anthropic": settings.llm.anthropic_default_model,
        "gemini": settings.llm.google_default_model,
        "ollama": "llama2",
        "deepseek": settings.llm.deepseek_default_model,
        "qwen": settings.llm.qwen_default_model,
        "xunfei": settings.llm.xunfei_default_model,
        "xai": settings.llm.xai_default_model,
    }
    
    def __init__(self):
        """初始化LLM提供商工厂"""
        self.provider_instances = {}  # 缓存已创建的提供商实例
        
        # 预配置的提供商配置
        self.provider_configs = {
            "openai": {
                "api_key": settings.llm.openai_api_key,
                "api_base": settings.llm.openai_api_base,
                "organization": settings.llm.openai_organization_id,
            },
            "anthropic": {
                "api_key": settings.llm.anthropic_api_key,
                "api_base": settings.llm.anthropic_api_url,
            },
            "azure_openai": {
                "api_key": settings.llm.azure_openai_api_key,
                "api_base": settings.llm.azure_openai_endpoint,
                "api_version": settings.llm.azure_openai_api_version,
            },
            "ollama": {
                "api_base": getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434"),
            },
            "gemini": {
                "api_key": settings.llm.google_api_key,
            },
            "deepseek": {
                "api_key": settings.llm.deepseek_api_key,
                "api_base": settings.llm.deepseek_api_base,
            },
            "qwen": {
                "api_key": settings.llm.qwen_api_key,
                "api_base": settings.llm.qwen_api_endpoint,
            },
            "xunfei": {
                "app_id": settings.llm.xunfei_app_id,
                "api_key": settings.llm.xunfei_api_key,
                "api_secret": settings.llm.xunfei_api_secret,
            },
            "xai": {
                "api_key": settings.llm.xai_api_key,
                "api_base": settings.llm.xai_api_base_url,
            },
        }
    
    async def create(self, provider_name: str, **kwargs) -> BaseLLMProvider:
        """创建LLM提供商实例
        
        Args:
            provider_name: 提供商名称
            **kwargs: 覆盖默认配置的参数
            
        Returns:
            BaseLLMProvider: LLM提供商实例
            
        Raises:
            FactoryException: 创建提供商实例失败时抛出异常
        """
        provider_name = provider_name.lower()
        
        # 检查提供商是否受支持
        if provider_name not in self.PROVIDER_MAPPING:
            raise FactoryException(f"不支持的LLM提供商: {provider_name}")
        
        # 创建缓存键，包含提供商名称和关键参数
        cache_key = provider_name
        if "api_key" in kwargs:
            # 如果提供了自定义API密钥，将其加入缓存键
            # 只取API键的前8个字符作为标识
            api_key_prefix = kwargs["api_key"][:8] if kwargs["api_key"] else "nokey"
            cache_key = f"{provider_name}_{api_key_prefix}"
        
        # 尝试从缓存中获取实例
        if cache_key in self.provider_instances:
            return self.provider_instances[cache_key]
        
        try:
            # 获取基础配置
            config = self.provider_configs.get(provider_name, {}).copy()
            
            # 使用传递的参数覆盖默认配置
            config.update(kwargs)
            
            # 动态导入提供商类
            provider_class_path = self.PROVIDER_MAPPING[provider_name]
            module_path, class_name = provider_class_path.rsplit(".", 1)
            
            try:
                module = importlib.import_module(module_path)
                provider_class = getattr(module, class_name)
            except (ImportError, AttributeError) as e:
                logger.error(f"导入LLM提供商类失败: {str(e)}")
                raise FactoryException(f"导入提供商类失败: {str(e)}")
            
            # 创建提供商实例
            provider_instance = provider_class(**config)
            
            # 缓存实例
            self.provider_instances[cache_key] = provider_instance
            
            return provider_instance
        except Exception as e:
            logger.error(f"创建LLM提供商实例失败: {str(e)}")
            raise FactoryException(f"创建提供商实例失败: {str(e)}")
    
    async def get_supported_providers(self) -> List[Dict[str, Any]]:
        """获取支持的LLM提供商列表
        
        Returns:
            List[Dict[str, Any]]: 支持的提供商列表，包含名称和默认模型
        """
        providers = []
        
        for provider_name, class_path in self.PROVIDER_MAPPING.items():
            # 检查提供商是否有配置
            config = self.provider_configs.get(provider_name, {})
            
            # 检查是否有API密钥（某些提供商可能不需要API密钥）
            has_api_key = "api_key" in config and config["api_key"] is not None
            
            # 获取默认模型
            default_model = self.DEFAULT_MODEL_MAPPING.get(provider_name, "")
            
            providers.append({
                "name": provider_name,
                "display_name": provider_name.replace("_", " ").title(),
                "default_model": default_model,
                "available": has_api_key or provider_name == "ollama"  # Ollama可能不需要API密钥
            })
        
        return providers
    
    def get_default_model(self, provider_name: str) -> str:
        """获取提供商的默认模型
        
        Args:
            provider_name: 提供商名称
            
        Returns:
            str: 默认模型名称
        """
        return self.DEFAULT_MODEL_MAPPING.get(provider_name.lower(), "")
