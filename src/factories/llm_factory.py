"""
LLM工厂模块

这个模块提供了创建不同LLM提供商模型的工厂类
"""

import logging
from typing import Dict, Any, List, Optional, Union
from src.config.settings import settings

# 定义基础类
class BaseChatModel:
    """Base class for chat models"""

    def invoke(self, *args, **kwargs):
        """Invoke the model"""
        raise NotImplementedError()
    
    async def generate_text(self, prompt: str, context: List[Dict], model: str, temperature: float = 0.7, max_tokens: int = 1000):
        """生成文本响应
        
        Args:
            prompt: 提示文本
            context: 上下文消息列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大生成令牌数
            
        Returns:
            str: 生成的文本
        """
        raise NotImplementedError()

# 尝试导入实际的模型
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    class ChatOpenAI(BaseChatModel):
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def invoke(self, *args, **kwargs):
            return {"content": "OpenAI模型未安装"}

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    class ChatAnthropic(BaseChatModel):
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def invoke(self, *args, **kwargs):
            return {"content": "Anthropic模型未安装"}

logger = logging.getLogger(__name__)


class LLMFactory:
    """LLM工厂类，负责创建不同的LLM模型实例"""
    
    def __init__(self):
        """初始化LLM工厂"""
        self.models = {}  # 缓存已创建的模型实例
        self.provider_configs = {
            "openai": {
                "api_key": settings.OPENAI_API_KEY,
                "base_url": settings.OPENAI_API_BASE,
                "organization": getattr(settings, "OPENAI_ORGANIZATION", None),
            },
            "anthropic": {
                "api_key": getattr(settings, "ANTHROPIC_API_KEY", ""),
            },
            "azure_openai": {
                "api_key": getattr(settings, "AZURE_OPENAI_API_KEY", ""),
                "base_url": getattr(settings, "AZURE_OPENAI_ENDPOINT", ""),
                "api_version": getattr(settings, "AZURE_OPENAI_API_VERSION", "2023-05-15"),
                "deployment_name": getattr(settings, "AZURE_OPENAI_DEPLOYMENT", ""),
            },
            "ollama": {
                "base_url": getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434"),
            },
            "gemini": {
                "api_key": getattr(settings, "GEMINI_API_KEY", ""),
            },
        }
    
    @staticmethod
    def create_from_config(provider: str, config: Dict[str, Any]) -> BaseChatModel:
        """
        根据提供商和配置创建LLM模型

        Args:
            provider: 模型提供商名称，如'openai', 'anthropic'等
            config: 模型配置，包括API密钥、模型名称等

        Returns:
            BaseChatModel: 聊天模型实例

        Raises:
            ValueError: 当提供商不支持或配置无效时
        """
        provider = provider.lower()

        # 从配置中提取通用参数
        temperature = config.get("temperature", 0.7)
        model_name = config.get("model_name", "")
        streaming = config.get("streaming", False)

        # 根据提供商创建对应的模型
        if provider == "openai":
            api_key = config.get("api_key", "")
            base_url = config.get("base_url", None)
            organization = config.get("organization", None)

            if not api_key and not base_url:
                raise ValueError("创建OpenAI模型需要提供API密钥或自定义base_url")

            model_kwargs = config.get("model_kwargs", {})

            return ChatOpenAI(
                model_name=model_name or "gpt-3.5-turbo",
                openai_api_key=api_key,
                openai_organization=organization,
                openai_api_base=base_url,
                temperature=temperature,
                streaming=streaming,
                model_kwargs=model_kwargs
            )

        elif provider == "anthropic":
            api_key = config.get("api_key", "")

            if not api_key:
                raise ValueError("创建Anthropic模型需要提供API密钥")

            return ChatAnthropic(
                model_name=model_name or "claude-2",
                anthropic_api_key=api_key,
                temperature=temperature,
                streaming=streaming
            )

        elif provider == "azure_openai":
            api_key = config.get("api_key", "")
            endpoint = config.get("base_url", "")
            api_version = config.get("api_version", "2023-05-15")
            deployment_name = config.get("deployment_name", "")

            if not api_key or not endpoint:
                raise ValueError("创建Azure OpenAI模型需要提供API密钥和终端点URL")

            # 使用Azure OpenAI
            try:
                from langchain_openai import AzureChatOpenAI
            except ImportError:
                class AzureChatOpenAI(BaseChatModel):
                    def __init__(self, **kwargs):
                        self.kwargs = kwargs

                    def invoke(self, *args, **kwargs):
                        return {"content": "Azure OpenAI模型未安装"}

            return AzureChatOpenAI(
                azure_endpoint=endpoint,
                azure_deployment=deployment_name or model_name,
                openai_api_key=api_key,
                openai_api_version=api_version,
                temperature=temperature,
                streaming=streaming
            )

        elif provider == "ollama":
            base_url = config.get("base_url", "http://localhost:11434")

            # 使用Ollama
            try:
                from langchain_community.chat_models import ChatOllama
            except ImportError:
                class ChatOllama(BaseChatModel):
                    def __init__(self, **kwargs):
                        self.kwargs = kwargs

                    def invoke(self, *args, **kwargs):
                        return {"content": "Ollama模型未安装"}

            return ChatOllama(
                model=model_name or "llama2",
                base_url=base_url,
                temperature=temperature
            )

        elif provider == "gemini":
            api_key = config.get("api_key", "")

            if not api_key:
                raise ValueError("创建Google Gemini模型需要提供API密钥")

            # 使用Google Gemini
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except ImportError:
                class ChatGoogleGenerativeAI(BaseChatModel):
                    def __init__(self, **kwargs):
                        self.kwargs = kwargs

                    def invoke(self, *args, **kwargs):
                        return {"content": "Google Gemini模型未安装"}

            return ChatGoogleGenerativeAI(
                model=model_name or "gemini-pro",
                google_api_key=api_key,
                temperature=temperature,
                streaming=streaming
            )

        else:
            raise ValueError(f"不支持的LLM提供商: {provider}")

    async def get_llm(self, provider: str) -> BaseChatModel:
        """
        获取LLM实例，如果已经缓存则直接返回，否则创建新实例
        
        Args:
            provider: 模型提供商名称
            
        Returns:
            BaseChatModel: 模型实例
            
        Raises:
            ValueError: 当提供商不支持或配置无效时
        """
        provider = provider.lower()
        
        # 检查缓存中是否已有该提供商的模型实例
        if provider in self.models:
            return self.models[provider]
        
        # 获取提供商的配置
        if provider not in self.provider_configs:
            raise ValueError(f"不支持的LLM提供商: {provider}")
        
        config = self.provider_configs[provider]
        
        # 创建模型实例
        model_instance = self.create_from_config(provider, config)
        
        # 将自定义的generate_text方法添加到原有模型实例
        if not hasattr(model_instance, 'generate_text'):
            async def generate_text(self, prompt: str, context: List[Dict], model: str = None, 
                                  temperature: float = 0.7, max_tokens: int = 1000):
                from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
                
                # 将context转换为消息列表
                messages = []
                for msg in context:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    
                    if role == "user":
                        messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        messages.append(AIMessage(content=content))
                    elif role == "system":
                        messages.append(SystemMessage(content=content))
                
                # 如果context为空或最后一条不是用户消息，添加当前prompt作为用户消息
                if not messages or not isinstance(messages[-1], HumanMessage):
                    messages.append(HumanMessage(content=prompt))
                
                # 调用模型
                try:
                    response = await self.ainvoke(messages)
                    if hasattr(response, 'content'):
                        return response.content
                    else:
                        return str(response)
                except Exception as e:
                    # 如果ainvoke不可用，尝试使用同步invoke
                    try:
                        response = self.invoke(messages)
                        if hasattr(response, 'content'):
                            return response.content
                        else:
                            return str(response)
                    except Exception as e2:
                        return f"模型调用出错: {str(e2)}"
            
            # 将方法绑定到实例
            import types
            model_instance.generate_text = types.MethodType(generate_text, model_instance)
        
        # 缓存模型实例
        self.models[provider] = model_instance
        
        return model_instance

    @staticmethod
    def get_supported_providers() -> List[str]:
        """
        获取支持的LLM提供商列表

        Returns:
            List[str]: 支持的提供商列表
        """
        return [
            "openai",
            "anthropic",
            "azure_openai",
            "ollama",
            "gemini"
        ]