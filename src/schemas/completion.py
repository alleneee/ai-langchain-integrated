"""
文本补全模式模块

该模块定义了文本补全相关的请求和响应模式。
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from src.schemas.base import UserBase

class CompletionRequest(UserBase):
    """文本补全请求"""
    inputs: Dict[str, Any] = Field(..., description="输入参数字典")
    response_mode: str = Field("blocking", description="响应模式：blocking或streaming")
    files: Optional[List[str]] = Field(None, description="文件ID列表")

class CompletionResponse(BaseModel):
    """文本补全响应"""
    id: str = Field(..., description="补全ID")
    text: str = Field(..., description="生成的文本")
    created_at: str = Field(..., description="创建时间") 