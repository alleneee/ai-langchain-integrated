"""
MCP工具实现示例模块

本模块提供了一些MCP工具的具体实现示例，展示如何创建和使用MCP工具
"""

from typing import Dict, Any, Optional, List
from langchain_mcp_adapters import MCPTool
from pydantic import BaseModel, Field

from .mcp_tools import mcp_tool_manager


class WebSearchInput(BaseModel):
    """
    网络搜索工具输入参数
    """
    search_term: str = Field(..., description="要搜索的关键词")
    explanation: Optional[str] = Field(None, description="搜索原因说明")


class SimpleWebSearchTool(MCPTool):
    """
    简单网络搜索工具实现示例
    """
    name: str = "web_search"
    description: str = "搜索互联网获取实时信息。当需要获取当前事件、技术更新或任何需要最新信息的主题时使用此工具。"
    input_schema: type = WebSearchInput
    
    def _execute(self, search_term: str, explanation: Optional[str] = None) -> str:
        """
        执行网络搜索
        
        Args:
            search_term: 搜索关键词
            explanation: 搜索原因说明（可选）
            
        Returns:
            搜索结果
        """
        # 实际实现中，这里会调用真实的搜索API
        # 这里仅作示例，返回一个模拟结果
        return f"搜索关键词 '{search_term}' 的结果示例。这里会展示实际搜索API返回的内容。"


class CalculatorInput(BaseModel):
    """
    计算器工具输入参数
    """
    expression: str = Field(..., description="要计算的数学表达式")


class SimpleCalculatorTool(MCPTool):
    """
    简单计算器工具实现示例
    """
    name: str = "calculator"
    description: str = "执行数学计算。当需要进行算术运算、解方程或其他数学操作时使用此工具。"
    input_schema: type = CalculatorInput
    
    def _execute(self, expression: str) -> str:
        """
        执行数学计算
        
        Args:
            expression: 数学表达式
            
        Returns:
            计算结果
        """
        try:
            # 警告：eval在生产环境中可能存在安全风险
            # 实际应用中应使用更安全的方式进行数学计算
            result = eval(expression)
            return f"计算结果: {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"


def register_example_tools() -> None:
    """
    注册示例工具到MCP工具管理器
    """
    tools = [
        SimpleWebSearchTool(),
        SimpleCalculatorTool()
    ]
    mcp_tool_manager.register_tools(tools)


# 提供一个示例函数，展示如何在应用启动时注册工具
def setup_mcp_tools() -> None:
    """
    设置并注册所有MCP工具
    
    在应用启动时调用此函数
    """
    # 注册示例工具
    register_example_tools()
    
    # 这里可以添加更多工具注册
    
    # 打印可用工具列表，用于调试
    tool_names = [schema["name"] for schema in mcp_tool_manager.get_tool_schemas()]
    print(f"已注册MCP工具: {', '.join(tool_names)}") 