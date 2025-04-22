from typing import Dict, Any, Optional, List
from langchain_core.tools import ToolException

# 尝试导入MCP适配器，如果不可用则使用模拟类
try:
    from langchain_mcp_adapters.client import MCPClient
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    # 创建模拟类
    class MCPClient:
        def __init__(self, api_base=None, api_key=None):
            self.api_base = api_base
            self.api_key = api_key

        async def invoke_tool(self, tool_name, parameters):
            return {"error": "MCP适配器未安装"}

        async def register_tools(self, tools):
            pass

class BaseMCPClient:
    """MCP 客户端基础封装

    封装langchain_mcp_adapters提供的MCPClient，提供统一的接口用于MCP工具的调用和注册。
    符合LangChain 0.3.x版本的接口规范。
    """

    def __init__(self, api_base: Optional[str] = None, api_key: Optional[str] = None):
        """初始化 MCP 客户端

        Args:
            api_base: MCP 服务的基础 URL
            api_key: MCP 服务的 API 密钥
        """
        self.client = MCPClient(api_base=api_base, api_key=api_key)

    async def invoke_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """统一的工具调用接口

        Args:
            tool_name: 要调用的工具名称
            parameters: 工具调用参数

        Returns:
            工具调用结果

        Raises:
            ToolException: 工具调用失败时抛出
        """
        try:
            return await self.client.invoke_tool(tool_name, parameters)
        except Exception as e:
            raise ToolException(f"工具 {tool_name} 调用失败: {str(e)}")

    async def register_tools(self, tools: List[Dict[str, Any]]) -> None:
        """注册工具到 MCP 服务

        Args:
            tools: 要注册的工具列表

        Raises:
            ToolException: 工具注册失败时抛出
        """
        try:
            await self.client.register_tools(tools)
        except Exception as e:
            raise ToolException(f"工具注册失败: {str(e)}")