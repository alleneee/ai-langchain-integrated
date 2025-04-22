"""
工作流API端点模块

提供工作流相关的API端点
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Dict, Any, Optional, List

from src.config.settings import get_settings

router = APIRouter()

class WorkflowResponse(BaseModel):
    """工作流响应模型"""
    message: str
    status: str

@router.get("/", response_model=WorkflowResponse)
async def get_workflow_status():
    """
    获取工作流状态
    
    Returns:
        WorkflowResponse: 工作流状态响应
    """
    return WorkflowResponse(
        message="工作流功能正在开发中",
        status="pending"
    )
