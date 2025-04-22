"""
工作流模式模块

该模块定义了工作流相关的请求和响应模式。
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from src.schemas.base import UserBase

class WorkflowRunRequest(UserBase):
    """工作流运行请求"""
    inputs: Dict[str, Any] = Field(..., description="输入参数字典")
    response_mode: str = Field("streaming", description="响应模式：blocking或streaming")

class WorkflowStopRequest(UserBase):
    """工作流停止请求"""
    task_id: str = Field(..., description="任务ID")

class WorkflowResult(BaseModel):
    """工作流结果"""
    id: str = Field(..., description="工作流运行ID")
    status: str = Field(..., description="运行状态")
    result: Optional[Dict[str, Any]] = Field(None, description="运行结果")
    created_at: str = Field(..., description="创建时间")
    completed_at: Optional[str] = Field(None, description="完成时间") 