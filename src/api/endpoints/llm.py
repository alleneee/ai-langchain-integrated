"""
LLM API接口模块

该模块实现了LLM相关的API接口。
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any

from src.schemas.llm import TokenEstimateRequest, TokenEstimateResponse, LLMProviderInfo
from src.schemas.responses import DataResponse
from src.services.llm_service import LLMService
from src.dependencies.common import get_settings_dependency

# 创建路由器
router = APIRouter()

# 服务实例
llm_service = LLMService()

@router.get("/providers", response_model=DataResponse[List[LLMProviderInfo]])
async def get_supported_providers():
    """
    获取支持的LLM提供商列表
    
    Returns:
        包含提供商列表的响应
    """
    try:
        providers = await llm_service.get_supported_providers()
        return DataResponse(data=providers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

@router.post("/estimate-tokens", response_model=DataResponse[TokenEstimateResponse])
async def estimate_tokens(request: TokenEstimateRequest):
    """
    估计文本的令牌数
    
    Args:
        request: 包含要估计的文本的请求
        
    Returns:
        包含令牌计数和字符计数的响应
    """
    try:
        result = await llm_service.estimate_token_count(request.text)
        response = TokenEstimateResponse(
            token_count=result["token_count"],
            character_count=result["character_count"]
        )
        return DataResponse(data=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}") 