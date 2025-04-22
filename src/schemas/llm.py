"""
LLM模式模块

该模块定义了LLM相关的请求和响应模式。
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

class TokenEstimateRequest(BaseModel):
    """令牌估计请求"""
    text: str

class TokenEstimateResponse(BaseModel):
    """令牌估计响应"""
    token_count: int
    character_count: int

class LLMProviderInfo(BaseModel):
    """LLM提供商信息"""
    name: str
    models: List[str]
    features: List[str] 