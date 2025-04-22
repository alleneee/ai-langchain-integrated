"""
工具API端点模块

提供MCP工具相关的API端点
"""

from typing import Dict, List, Any
from fastapi import APIRouter, HTTPException, Depends

from src.utils.mcp_tools import mcp_tool_manager, get_available_tool_names
from src.schemas.common import BaseResponse

router = APIRouter()


@router.get("/tools", response_model=BaseResponse[List[str]])
async def list_tools() -> BaseResponse[List[str]]:
    """
    获取所有可用工具列表
    
    Returns:
        包含工具名称列表的响应
    """
    tool_names = get_available_tool_names()
    return BaseResponse(
        data=tool_names,
        message="成功获取工具列表",
    )


@router.get("/tools/schemas", response_model=BaseResponse[List[Dict[str, Any]]])
async def get_tool_schemas() -> BaseResponse[List[Dict[str, Any]]]:
    """
    获取所有工具的详细信息
    
    Returns:
        包含工具详细信息的响应
    """
    schemas = mcp_tool_manager.get_tool_schemas()
    return BaseResponse(
        data=schemas,
        message="成功获取工具详细信息",
    )


@router.post("/tools/{tool_name}/execute", response_model=BaseResponse[Any])
async def execute_tool(tool_name: str, tool_params: Dict[str, Any]) -> BaseResponse[Any]:
    """
    执行指定工具
    
    Args:
        tool_name: 工具名称
        tool_params: 工具执行参数
        
    Returns:
        工具执行结果
        
    Raises:
        HTTPException: 如果工具不存在或执行出错
    """
    try:
        result = mcp_tool_manager.execute_tool(tool_name, **tool_params)
        return BaseResponse(
            data=result,
            message=f"成功执行工具 '{tool_name}'",
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"工具执行错误: {str(e)}") 