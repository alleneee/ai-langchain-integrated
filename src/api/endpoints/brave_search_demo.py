"""
Brave Search MCP 演示API端点模块

提供用于演示和测试 Brave Search MCP 功能的API端点
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from src.schemas.common import BaseResponse
from src.config.brave_search_config import brave_search_settings
from src.utils.brave_search_tools import brave_search_manager, BraveWebSearchTool, BraveNewsSearchTool, BraveImageSearchTool

router = APIRouter()


class SearchRequest(BaseModel):
    """
    搜索请求模型
    """
    query: str = Field(..., description="搜索查询关键词")
    max_results: Optional[int] = Field(10, description="返回结果的最大数量")


@router.get("/status", response_model=BaseResponse[Dict[str, Any]])
async def get_brave_search_status() -> BaseResponse[Dict[str, Any]]:
    """
    获取 Brave Search MCP 服务器状态
    
    Returns:
        包含服务器状态信息的响应
    """
    status = {
        "enabled": brave_search_settings.enabled,
        "running": brave_search_manager.is_running if brave_search_settings.enabled else False,
        "config": {
            "server_key": f"{brave_search_settings.server_key[:4]}...{brave_search_settings.server_key[-4:]}",  # 部分隐藏密钥
            "max_results": brave_search_settings.max_results,
            "timeout": brave_search_settings.timeout,
            "search_language": brave_search_settings.search_language,
            "search_region": brave_search_settings.search_region
        }
    }
    
    return BaseResponse(
        data=status,
        message="成功获取 Brave Search MCP 状态",
    )


@router.post("/web", response_model=BaseResponse[List[Dict[str, Any]]])
async def search_web(request: SearchRequest) -> BaseResponse[List[Dict[str, Any]]]:
    """
    执行网页搜索
    
    Args:
        request: 搜索请求
        
    Returns:
        搜索结果
    """
    if not brave_search_settings.enabled:
        raise HTTPException(status_code=400, detail="Brave Search MCP 已禁用")
        
    if not brave_search_manager.is_running:
        raise HTTPException(status_code=400, detail="Brave Search MCP 服务器未运行")
    
    # 使用 Brave Web 搜索工具
    search_tool = BraveWebSearchTool()
    results = await search_tool._aexecute(
        search_term=request.query,
        max_results=request.max_results
    )
    
    return BaseResponse(
        data=results,
        message=f"成功搜索网页: {request.query}",
    )


@router.post("/news", response_model=BaseResponse[List[Dict[str, Any]]])
async def search_news(request: SearchRequest) -> BaseResponse[List[Dict[str, Any]]]:
    """
    执行新闻搜索
    
    Args:
        request: 搜索请求
        
    Returns:
        搜索结果
    """
    if not brave_search_settings.enabled:
        raise HTTPException(status_code=400, detail="Brave Search MCP 已禁用")
        
    if not brave_search_manager.is_running:
        raise HTTPException(status_code=400, detail="Brave Search MCP 服务器未运行")
    
    # 使用 Brave 新闻搜索工具
    search_tool = BraveNewsSearchTool()
    results = await search_tool._aexecute(
        search_term=request.query,
        max_results=request.max_results
    )
    
    return BaseResponse(
        data=results,
        message=f"成功搜索新闻: {request.query}",
    )


@router.post("/images", response_model=BaseResponse[List[Dict[str, Any]]])
async def search_images(request: SearchRequest) -> BaseResponse[List[Dict[str, Any]]]:
    """
    执行图片搜索
    
    Args:
        request: 搜索请求
        
    Returns:
        搜索结果
    """
    if not brave_search_settings.enabled:
        raise HTTPException(status_code=400, detail="Brave Search MCP 已禁用")
        
    if not brave_search_manager.is_running:
        raise HTTPException(status_code=400, detail="Brave Search MCP 服务器未运行")
    
    # 使用 Brave 图片搜索工具
    search_tool = BraveImageSearchTool()
    results = await search_tool._aexecute(
        search_term=request.query,
        max_results=request.max_results
    )
    
    return BaseResponse(
        data=results,
        message=f"成功搜索图片: {request.query}",
    )


@router.post("/restart", response_model=BaseResponse[Dict[str, Any]])
async def restart_brave_search_server() -> BaseResponse[Dict[str, Any]]:
    """
    重启 Brave Search MCP 服务器
    
    Returns:
        重启结果
    """
    if not brave_search_settings.enabled:
        raise HTTPException(status_code=400, detail="Brave Search MCP 已禁用")
    
    # 停止现有服务器
    brave_search_manager.stop_server()
    
    # 启动新服务器
    try:
        await brave_search_manager.start_server()
        return BaseResponse(
            data={"status": "restarted"},
            message="Brave Search MCP 服务器已重启",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"重启 Brave Search MCP 服务器失败: {str(e)}"
        ) 