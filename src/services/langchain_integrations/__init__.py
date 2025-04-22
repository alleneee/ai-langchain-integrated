"""
LangChain 集成模块

提供与不同 LLM 服务商的 LangChain 集成
"""

from src.services.langchain_integrations.base import BaseLangChainIntegration
from src.services.langchain_integrations.langchain_deepseek import DeepSeekIntegration
from src.services.langchain_integrations.langchain_xai import XAIIntegration
from src.services.langchain_integrations.langchain_gemini import GeminiIntegration
from src.services.langchain_integrations.langchain_qwen import QwenIntegration

# 提供商映射
INTEGRATIONS = {
    "deepseek": DeepSeekIntegration,
    "xai": XAIIntegration,
    "gemini": GeminiIntegration,
    "qwen": QwenIntegration
}

__all__ = [
    "BaseLangChainIntegration",
    "DeepSeekIntegration",
    "XAIIntegration",
    "GeminiIntegration",
    "QwenIntegration",
    "INTEGRATIONS"
]
