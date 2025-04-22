from fastapi import Depends, Request
from typing import Dict, Any, Optional

from src.utils.langchain_mcp_client import BaseMCPClient
from src.utils.mcp_tools import PlaywrightMCPTool, BraveSearchMCPTool

def get_mcp_client(request: Request) -> BaseMCPClient:
    """获取 MCP 客户端实例
    
    Args:
        request: FastAPI 请求对象
        
    Returns:
        MCP 客户端实例
    """
    return request.app.state.mcp_client

def get_playwright_tool(request: Request) -> Optional[PlaywrightMCPTool]:
    """获取 Playwright 工具实例
    
    Args:
        request: FastAPI 请求对象
        
    Returns:
        Playwright 工具实例，如果未初始化则返回 None
    """
    return getattr(request.app.state, "playwright_tool", None)

def get_brave_search_tool(request: Request) -> Optional[BraveSearchMCPTool]:
    """获取 Brave Search 工具实例
    
    Args:
        request: FastAPI 请求对象
        
    Returns:
        Brave Search 工具实例，如果未初始化则返回 None
    """
    return getattr(request.app.state, "brave_search_tool", None)

async def invoke_mcp_tool(
    tool_name: str, 
    parameters: Dict[str, Any],
    client: BaseMCPClient = Depends(get_mcp_client)
) -> Dict[str, Any]:
    """通用工具调用依赖函数
    
    Args:
        tool_name: 工具名称
        parameters: 工具参数
        client: MCP 客户端实例
        
    Returns:
        工具调用结果
    """
    try:
        return await client.invoke_tool(tool_name, parameters)
    except Exception as e:
        return {"error": f"工具调用失败: {str(e)}"} 