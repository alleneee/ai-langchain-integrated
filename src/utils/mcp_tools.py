"""
MCP工具类模块，提供与MCP服务交互的工具函数
基于langchain-mcp-adapters包实现
"""

from typing import List, Dict, Any, Optional, Callable, Awaitable
from langchain_core.tools import BaseTool, tool

# 尝试导入MCP适配器，如果不可用则使用模拟类
try:
    from langchain_mcp_adapters import MCPToolRegistry, MCPTool
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    # 创建模拟类
    class MCPToolRegistry:
        def __init__(self):
            self.tools = {}

        def register_tool(self, tool):
            self.tools[tool.name] = tool

        def get_all_langchain_tools(self):
            return []

        def get_tool_by_name(self, name):
            return self.tools.get(name)

        def get_all_tool_schemas(self):
            return []

    class MCPTool:
        def __init__(self):
            self.name = ""
            self.description = ""

        def as_langchain_tool(self):
            return None

        def execute(self, **kwargs):
            return {"error": "MCP适配器未安装"}

from src.config.playwright_config import playwright_settings
from src.config.brave_search_config import brave_search_settings


class MCPToolManager:
    """
    MCP工具管理器

    用于注册和管理MCP工具，提供工具的获取和转换功能
    """

    def __init__(self):
        """
        初始化MCP工具管理器
        """
        self.registry = MCPToolRegistry()

    def register_tools(self, tools: List[MCPTool]) -> None:
        """
        注册MCP工具到注册表

        Args:
            tools: 要注册的MCP工具列表
        """
        for tool in tools:
            self.registry.register_tool(tool)

    def get_all_tools(self) -> List[BaseTool]:
        """
        获取所有注册的MCP工具，转换为LangChain工具

        Returns:
            所有MCP工具转换后的LangChain工具列表
        """
        return self.registry.get_all_langchain_tools()

    def get_tools_by_names(self, tool_names: List[str]) -> List[BaseTool]:
        """
        根据工具名获取MCP工具，转换为LangChain工具

        Args:
            tool_names: 工具名列表

        Returns:
            指定工具名的LangChain工具列表
        """
        tools = []
        for name in tool_names:
            tool = self.registry.get_tool_by_name(name)
            if tool:
                tools.append(tool.as_langchain_tool())
        return tools

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        获取所有注册工具的schema描述

        Returns:
            所有工具的schema描述列表
        """
        return self.registry.get_all_tool_schemas()

    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """
        直接执行指定的工具

        Args:
            tool_name: 工具名称
            **kwargs: 工具执行参数

        Returns:
            工具执行结果

        Raises:
            ValueError: 如果工具不存在
        """
        tool = self.registry.get_tool_by_name(tool_name)
        if not tool:
            raise ValueError(f"工具 '{tool_name}' 不存在")

        return tool.execute(**kwargs)


# 创建单例实例，方便全局使用
mcp_tool_manager = MCPToolManager()


def register_default_tools() -> None:
    """
    注册默认的MCP工具
    """
    # 这里可以添加项目中常用的MCP工具
    pass


def get_available_tool_names() -> List[str]:
    """
    获取所有可用的MCP工具名称

    Returns:
        所有注册工具的名称列表
    """
    return [schema["name"] for schema in mcp_tool_manager.get_tool_schemas()]


class PlaywrightMCPTool(MCPTool if MCP_AVAILABLE else object):
    """Playwright MCP 工具封装"""

    name: str = "playwright"
    description: str = "使用 Playwright 进行网页自动化操作"
    args_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["navigate", "screenshot", "click", "type"],
                "description": "要执行的操作类型"
            },
            "url": {
                "type": "string",
                "description": "导航的目标URL，仅在action为navigate时使用"
            },
            "selector": {
                "type": "string",
                "description": "CSS选择器，仅在action为click或type时使用"
            },
            "text": {
                "type": "string",
                "description": "要输入的文本，仅在action为type时使用"
            }
        },
        "required": ["action"]
    }

    def __init__(self):
        """初始化 Playwright 工具"""
        super().__init__()
        self.browser = None
        self.page = None

    async def initialize(self) -> None:
        """初始化 Playwright 资源"""
        from playwright.async_api import async_playwright

        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=playwright_settings.headless
        )
        self.page = await self.browser.new_page()

    @tool
    async def _run(self, **kwargs: Any) -> Any:
        """使用 Playwright 进行网页自动化操作

        Args:
            action: 要执行的操作类型（navigate, screenshot, click, type）
            url: 导航的目标URL，仅在action为navigate时使用
            selector: CSS选择器，仅在action为click或type时使用
            text: 要输入的文本，仅在action为type时使用

        Returns:
            操作结果
        """
        action = kwargs.get("action")
        if action == "navigate":
            return await self._navigate(kwargs.get("url"))
        elif action == "screenshot":
            return await self._take_screenshot()
        elif action == "click":
            return await self._click(kwargs.get("selector"))
        elif action == "type":
            return await self._type(kwargs.get("selector"), kwargs.get("text"))
        else:
            return {"error": f"不支持的操作: {action}"}

    async def _navigate(self, url: str) -> Dict[str, Any]:
        """导航到指定 URL

        Args:
            url: 目标 URL

        Returns:
            导航结果
        """
        if not self.page:
            return {"error": "浏览器未初始化"}

        await self.page.goto(url)
        return {
            "status": "success",
            "title": await self.page.title(),
            "url": self.page.url
        }

    async def _take_screenshot(self) -> Dict[str, Any]:
        """截取页面截图

        Returns:
            截图数据（Base64 编码）
        """
        if not self.page:
            return {"error": "浏览器未初始化"}

        screenshot = await self.page.screenshot(type="jpeg", quality=80)
        import base64
        encoded = base64.b64encode(screenshot).decode("utf-8")
        return {
            "status": "success",
            "data": encoded,
            "mime_type": "image/jpeg"
        }

    async def _click(self, selector: str) -> Dict[str, Any]:
        """点击页面元素

        Args:
            selector: CSS 选择器

        Returns:
            点击结果
        """
        if not self.page:
            return {"error": "浏览器未初始化"}

        try:
            await self.page.click(selector)
            return {"status": "success"}
        except Exception as e:
            return {"error": f"点击失败: {str(e)}"}

    async def _type(self, selector: str, text: str) -> Dict[str, Any]:
        """在页面元素中输入文本

        Args:
            selector: CSS 选择器
            text: 输入文本

        Returns:
            输入结果
        """
        if not self.page:
            return {"error": "浏览器未初始化"}

        try:
            await self.page.fill(selector, text)
            return {"status": "success"}
        except Exception as e:
            return {"error": f"输入失败: {str(e)}"}

    async def shutdown(self) -> None:
        """关闭 Playwright 资源"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()


class BraveSearchMCPTool(MCPTool if MCP_AVAILABLE else object):
    """Brave Search MCP 工具封装"""

    name: str = "brave_search"
    description: str = "使用 Brave Search 进行网络搜索"
    args_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索查询"
            }
        },
        "required": ["query"]
    }

    def __init__(self):
        """初始化 Brave Search 工具"""
        super().__init__()
        self.api_key = brave_search_settings.api_key
        self.api_base = brave_search_settings.api_base

    async def initialize(self) -> None:
        """初始化 Brave Search 资源"""
        import aiohttp
        self.session = aiohttp.ClientSession()

    @tool
    async def _run(self, **kwargs: Any) -> Any:
        """使用 Brave Search 进行网络搜索

        Args:
            query: 搜索查询

        Returns:
            搜索结果
        """
        query = kwargs.get("query")
        if not query:
            return {"error": "缺少查询参数"}

        return await self._search(query)

    async def _search(self, query: str) -> Dict[str, Any]:
        """执行搜索查询

        Args:
            query: 搜索查询

        Returns:
            搜索结果
        """
        if not self.session:
            return {"error": "会话未初始化"}

        try:
            headers = {
                "X-Subscription-Token": self.api_key,
                "Accept": "application/json"
            }
            async with self.session.get(
                f"{self.api_base}/search",
                params={"q": query, "count": 10},
                headers=headers
            ) as response:
                if response.status != 200:
                    return {"error": f"搜索失败: HTTP {response.status}"}

                data = await response.json()
                return {
                    "status": "success",
                    "results": data.get("web", {}).get("results", [])
                }
        except Exception as e:
            return {"error": f"搜索失败: {str(e)}"}

    async def shutdown(self) -> None:
        """关闭 Brave Search 资源"""
        if self.session:
            await self.session.close()