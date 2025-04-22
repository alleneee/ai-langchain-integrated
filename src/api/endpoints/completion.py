"""
补全API端点模块

提供文本补全相关的API端点
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Dict, Any, Optional, List

from src.config.settings import get_settings
from src.services.completion_service import CompletionService

router = APIRouter()

class CompletionRequest(BaseModel):
    """补全请求模型"""
    prompt: str
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    
class CompletionResponse(BaseModel):
    """补全响应模型"""
    text: str
    model: str
    
@router.post("/", response_model=CompletionResponse)
async def create_completion(
    request: CompletionRequest,
    settings = Depends(get_settings)
):
    """
    创建文本补全
    
    Args:
        request: 补全请求
        
    Returns:
        CompletionResponse: 补全响应
    """
    try:
        # 使用默认设置
        model = request.model or settings.DEFAULT_CHAT_MODEL
        max_tokens = request.max_tokens or settings.DEFAULT_MAX_TOKENS
        temperature = request.temperature or settings.DEFAULT_TEMPERATURE
        
        # 创建补全服务
        completion_service = CompletionService(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # 获取补全结果
        result = await completion_service.create_completion(request.prompt)
        
        return CompletionResponse(
            text=result,
            model=model
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"补全生成失败: {str(e)}"
        )
