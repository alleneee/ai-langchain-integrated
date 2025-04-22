"""
Playwright MCP 演示API端点模块

提供用于演示和测试 Playwright MCP 功能的API端点
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from src.schemas.common import BaseResponse
from src.config.playwright_config import playwright_settings
from src.utils.playwright_tools import playwright_mcp_manager

router = APIRouter()


class NavigateRequest(BaseModel):
    """
    导航请求模型
    """
    url: str = Field(..., description="要导航到的URL")


class ScreenshotResponse(BaseModel):
    """
    截图响应模型
    """
    image_base64: str = Field(..., description="Base64编码的截图数据")
    image_type: str = Field("jpeg", description="图像类型")


@router.get("/status", response_model=BaseResponse[Dict[str, Any]])
async def get_playwright_status() -> BaseResponse[Dict[str, Any]]:
    """
    获取 Playwright MCP 服务器状态
    
    Returns:
        包含服务器状态信息的响应
    """
    status = {
        "enabled": playwright_settings.enabled,
        "running": playwright_mcp_manager.is_running if playwright_settings.enabled else False,
        "config": {
            "headless": playwright_settings.headless,
            "port": playwright_settings.port,
            "snapshot_mode": playwright_settings.snapshot_mode,
            "browser": playwright_settings.browser,
            "viewport": {
                "width": playwright_settings.viewport_width,
                "height": playwright_settings.viewport_height
            }
        }
    }
    
    return BaseResponse(
        data=status,
        message="成功获取 Playwright MCP 状态",
    )


@router.post("/navigate", response_model=BaseResponse[Dict[str, Any]])
async def navigate_to_url(
    request: NavigateRequest, 
    background_tasks: BackgroundTasks
) -> BaseResponse[Dict[str, Any]]:
    """
    导航到指定URL
    
    Args:
        request: 导航请求
        background_tasks: 后台任务
        
    Returns:
        导航结果
    """
    if not playwright_settings.enabled:
        raise HTTPException(status_code=400, detail="Playwright MCP 已禁用")
        
    if not playwright_mcp_manager.is_running:
        raise HTTPException(status_code=400, detail="Playwright MCP 服务器未运行")
    
    # 这里应该实际调用 Playwright MCP API 执行导航
    # 在实际实现中，需要集成与 Playwright MCP 服务器的通信
    
    return BaseResponse(
        data={"url": request.url, "status": "success"},
        message=f"已导航到 {request.url}",
    )


@router.get("/screenshot", response_model=BaseResponse[ScreenshotResponse])
async def take_screenshot(raw: bool = False) -> BaseResponse[ScreenshotResponse]:
    """
    获取当前页面的截图
    
    Args:
        raw: 是否返回未压缩图像
        
    Returns:
        包含截图数据的响应
    """
    if not playwright_settings.enabled:
        raise HTTPException(status_code=400, detail="Playwright MCP 已禁用")
        
    if not playwright_mcp_manager.is_running:
        raise HTTPException(status_code=400, detail="Playwright MCP 服务器未运行")
    
    # 这里应该实际调用 Playwright MCP API 获取截图
    # 在实际实现中，需要集成与 Playwright MCP 服务器的通信
    
    # 模拟截图响应
    image_type = "png" if raw else "jpeg"
    
    return BaseResponse(
        data=ScreenshotResponse(
            image_base64="模拟的Base64编码截图数据",
            image_type=image_type
        ),
        message="成功获取页面截图",
    )


@router.post("/restart", response_model=BaseResponse[Dict[str, Any]])
async def restart_playwright_server() -> BaseResponse[Dict[str, Any]]:
    """
    重启 Playwright MCP 服务器
    
    Returns:
        重启结果
    """
    if not playwright_settings.enabled:
        raise HTTPException(status_code=400, detail="Playwright MCP 已禁用")
    
    # 停止现有服务器
    playwright_mcp_manager.stop_server()
    
    # 启动新服务器
    try:
        await playwright_mcp_manager.start_server()
        return BaseResponse(
            data={"status": "restarted"},
            message="Playwright MCP 服务器已重启",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"重启 Playwright MCP 服务器失败: {str(e)}"
        ) 