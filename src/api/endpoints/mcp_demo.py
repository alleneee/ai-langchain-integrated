"""
MCP 工具演示端点

该模块提供了 MCP 工具演示 API 端点。
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, List, Optional
from langchain_core.tools import ToolException

from src.utils.langchain_mcp_client import BaseMCPClient
from src.dependencies.tools import get_mcp_client, get_playwright_tool, get_brave_search_tool
from src.factories.mcp_tool_factory import MCPToolFactory

router = APIRouter()

@router.get("/tools", summary="获取可用的 MCP 工具列表")
async def list_mcp_tools() -> Dict[str, List[str]]:
    """获取所有已注册的 MCP 工具列表"""
    tools = MCPToolFactory.get_registered_tools()
    return {"tools": tools}

@router.post("/playwright", summary="调用 Playwright MCP 工具")
async def call_playwright_tool(
    action: str,
    url: Optional[str] = None,
    selector: Optional[str] = None,
    text: Optional[str] = None,
    client: BaseMCPClient = Depends(get_mcp_client)
) -> Dict[str, Any]:
    """调用 Playwright MCP 工具
    
    参数:
        action: 操作类型（navigate, screenshot, click, type）
        url: 导航 URL，仅在 action 为 navigate 时使用
        selector: CSS 选择器，仅在 action 为 click 或 type 时使用
        text: 输入文本，仅在 action 为 type 时使用
        client: MCP 客户端实例
        
    返回:
        工具调用结果
    """
    params: Dict[str, Any] = {"action": action}
    
    if action == "navigate" and url:
        params["url"] = url
    elif action == "click" and selector:
        params["selector"] = selector
    elif action == "type" and selector and text:
        params["selector"] = selector
        params["text"] = text
    
    try:
        result = await client.invoke_tool("playwright", params)
        return result
    except ToolException as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search", summary="调用 Brave Search MCP 工具")
async def call_search_tool(
    query: str,
    client: BaseMCPClient = Depends(get_mcp_client)
) -> Dict[str, Any]:
    """调用 Brave Search MCP 工具
    
    参数:
        query: 搜索查询
        client: MCP 客户端实例
        
    返回:
        搜索结果
    """
    try:
        result = await client.invoke_tool("brave_search", {"query": query})
        return result
    except ToolException as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status", summary="获取 MCP 工具状态")
async def get_mcp_status(
    playwright_tool = Depends(get_playwright_tool),
    brave_search_tool = Depends(get_brave_search_tool)
) -> Dict[str, Any]:
    """获取 MCP 工具状态"""
    return {
        "playwright": {
            "enabled": playwright_tool is not None,
            "initialized": hasattr(playwright_tool, "page") and playwright_tool.page is not None
        },
        "brave_search": {
            "enabled": brave_search_tool is not None,
            "initialized": hasattr(brave_search_tool, "session") and brave_search_tool.session is not None
        }
    } 