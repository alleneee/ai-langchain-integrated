from typing import Dict, Type, Any, Optional, List
from langchain_core.tools import ToolException

# 尝试导入MCP适配器，如果不可用则使用模拟类
try:
    from langchain_mcp_adapters.tools import MCPTool
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    # 创建模拟类
    class MCPTool:
        def __init__(self):
            self.name = ""
            self.description = ""

from src.utils.mcp_tools import PlaywrightMCPTool, BraveSearchMCPTool

class MCPToolFactory:
    """MCP 工具工厂

    用于创建和管理 MCP 工具实例，支持注册自定义工具类型。
    """

    _tools: Dict[str, Type[MCPTool]] = {
        "playwright": PlaywrightMCPTool,
        "brave_search": BraveSearchMCPTool,
    }

    @classmethod
    def create(cls, tool_type: str, **kwargs: Any) -> Optional[MCPTool]:
        """创建指定类型的 MCP 工具实例

        Args:
            tool_type: 工具类型名称
            **kwargs: 传递给工具构造函数的额外参数

        Returns:
            创建的工具实例，如果类型未注册则返回 None

        Raises:
            ToolException: 如果工具类型未注册或创建失败
        """
        if tool_type not in cls._tools:
            raise ToolException(f"未知的工具类型: {tool_type}")

        tool_class = cls._tools[tool_type]
        try:
            return tool_class(**kwargs)
        except Exception as e:
            raise ToolException(f"创建工具 {tool_type} 失败: {str(e)}")

    @classmethod
    def register_tool(cls, tool_type: str, tool_class: Type[MCPTool]) -> None:
        """注册新的工具类型

        Args:
            tool_type: 工具类型名称
            tool_class: 工具类
        """
        cls._tools[tool_type] = tool_class

    @classmethod
    def get_registered_tools(cls) -> List[str]:
        """获取所有已注册的工具类型

        Returns:
            工具类型名称列表
        """
        return list(cls._tools.keys())

    @classmethod
    def create_all(cls, **kwargs: Any) -> Dict[str, MCPTool]:
        """创建所有已注册的工具实例

        Args:
            **kwargs: 传递给各工具构造函数的额外参数

        Returns:
            工具实例字典，键为工具类型名称，值为工具实例
        """
        tools = {}
        for tool_type in cls._tools:
            try:
                tools[tool_type] = cls.create(tool_type, **kwargs)
            except ToolException as e:
                print(f"创建工具 {tool_type} 失败: {str(e)}")
        return tools