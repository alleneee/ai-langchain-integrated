"""
嵌入模型工厂模块

这个模块提供了创建不同嵌入模型的工厂类
"""

import logging
from typing import Dict, Any, List

# 定义基础类
class Embeddings:
    """Base class for embeddings"""

    def embed_documents(self, texts):
        """Embed search docs."""
        raise NotImplementedError()

    def embed_query(self, text):
        """Embed query text."""
        raise NotImplementedError()

# 尝试导入实际的嵌入模型
try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    class OpenAIEmbeddings(Embeddings):
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def embed_documents(self, texts):
            return [[0.0] * 1536] * len(texts)

        def embed_query(self, text):
            return [0.0] * 1536

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    class HuggingFaceEmbeddings(Embeddings):
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def embed_documents(self, texts):
            return [[0.0] * 384] * len(texts)

        def embed_query(self, text):
            return [0.0] * 384

logger = logging.getLogger(__name__)


class EmbeddingFactory:
    """嵌入模型工厂类，负责创建不同的嵌入模型实例"""

    @staticmethod
    def create_from_config(provider: str, config: Dict[str, Any]) -> Embeddings:
        """
        根据提供商和配置创建嵌入模型

        Args:
            provider: 嵌入模型提供商名称，如'openai', 'huggingface'等
            config: 嵌入模型配置，包括API密钥、模型名称等

        Returns:
            Embeddings: 嵌入模型实例

        Raises:
            ValueError: 当提供商不支持或配置无效时
        """
        provider = provider.lower()

        # 根据提供商创建对应的嵌入模型
        if provider == "openai":
            api_key = config.get("api_key", "")
            base_url = config.get("base_url", None)
            organization = config.get("organization", None)
            model_name = config.get("model_name", "text-embedding-3-small")

            if not api_key and not base_url:
                raise ValueError("创建OpenAI嵌入模型需要提供API密钥或自定义base_url")

            return OpenAIEmbeddings(
                model=model_name,
                openai_api_key=api_key,
                openai_organization=organization,
                openai_api_base=base_url
            )

        elif provider == "azure_openai":
            api_key = config.get("api_key", "")
            endpoint = config.get("base_url", "")
            api_version = config.get("api_version", "2023-05-15")
            deployment_name = config.get("deployment_name", "")
            model_name = config.get("model_name", "")

            if not api_key or not endpoint:
                raise ValueError("创建Azure OpenAI嵌入模型需要提供API密钥和终端点URL")

            # 使用Azure OpenAI的嵌入模型
            try:
                from langchain_openai import AzureOpenAIEmbeddings
            except ImportError:
                # 如果导入失败，使用模拟类
                class AzureOpenAIEmbeddings(Embeddings):
                    def __init__(self, **kwargs):
                        self.kwargs = kwargs

                    def embed_documents(self, texts):
                        return [[0.0] * 1536] * len(texts)

                    def embed_query(self, text):
                        return [0.0] * 1536

            return AzureOpenAIEmbeddings(
                azure_endpoint=endpoint,
                azure_deployment=deployment_name or model_name,
                openai_api_key=api_key,
                openai_api_version=api_version
            )

        elif provider == "huggingface":
            model_name = config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")

            return HuggingFaceEmbeddings(
                model_name=model_name
            )

        elif provider == "ollama":
            base_url = config.get("base_url", "http://localhost:11434")
            model_name = config.get("model_name", "llama2")

            # 使用Ollama嵌入模型
            try:
                from langchain_community.embeddings import OllamaEmbeddings
            except ImportError:
                # 如果导入失败，使用模拟类
                class OllamaEmbeddings(Embeddings):
                    def __init__(self, **kwargs):
                        self.kwargs = kwargs

                    def embed_documents(self, texts):
                        return [[0.0] * 768] * len(texts)

                    def embed_query(self, text):
                        return [0.0] * 768

            return OllamaEmbeddings(
                model_name=model_name,
                base_url=base_url
            )

        else:
            raise ValueError(f"不支持的嵌入模型提供商: {provider}")

    @staticmethod
    def get_supported_providers() -> List[str]:
        """
        获取支持的嵌入模型提供商列表

        Returns:
            List[str]: 支持的提供商列表
        """
        return [
            "openai",
            "azure_openai",
            "huggingface",
            "ollama"
        ]