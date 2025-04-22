"""
Playwright MCP 工具实现模块

本模块提供了基于 Playwright MCP 的工具实现，用于自动化浏览器操作
"""

import os
import json
import asyncio
import subprocess
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field

from langchain_core.callbacks import CallbackManagerForToolRun
from .mcp_tools import mcp_tool_manager, MCP_AVAILABLE

# 尝试导入MCP适配器，如果不可用则使用模拟类
try:
    from langchain_mcp_adapters import MCPTool
except ImportError:
    # 创建模拟类
    class MCPTool:
        def __init__(self):
            self.name = ""
            self.description = ""
            self.input_schema = None

        async def _aexecute(self, *args, **kwargs):
            return "MCP适配器未安装"

# 导入 Playwright 配置
from src.config.playwright_config import playwright_settings


class PlaywrightMCPManager:
    """
    Playwright MCP 管理器

    用于启动、停止 Playwright MCP 服务器并管理与其的通信
    """
    _instance = None
    _initialized = False
    _mcp_process = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PlaywrightMCPManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not PlaywrightMCPManager._initialized:
            self._is_running = False
            PlaywrightMCPManager._initialized = True

    async def start_server(self):
        """
        启动 Playwright MCP 服务器
        """
        if not playwright_settings.enabled:
            print("Playwright MCP 已禁用，跳过服务器启动")
            return

        if self._is_running:
            return

        # 构建命令行参数
        args = ["npx", "@playwright/mcp@latest"]

        # 添加选项
        if playwright_settings.headless:
            args.append("--headless")
        if not playwright_settings.snapshot_mode:
            args.append("--vision")
        args.extend(["--port", str(playwright_settings.port)])

        # 启动进程
        try:
            self._mcp_process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            print(f"Playwright MCP 服务器已启动，端口: {playwright_settings.port}")

            # 等待服务器启动
            await asyncio.sleep(2)
            self._is_running = True
        except Exception as e:
            print(f"启动 Playwright MCP 服务器失败: {str(e)}")
            raise

    def stop_server(self):
        """
        停止 Playwright MCP 服务器
        """
        if not self._is_running or self._mcp_process is None:
            return

        try:
            self._mcp_process.terminate()
            self._mcp_process.wait(timeout=5)
            self._is_running = False
            print("Playwright MCP 服务器已停止")
        except Exception as e:
            print(f"停止 Playwright MCP 服务器失败: {str(e)}")
            # 尝试强制终止
            try:
                self._mcp_process.kill()
            except:
                pass
            self._is_running = False

    @property
    def is_running(self):
        """
        检查服务器是否正在运行

        Returns:
            bool: 服务器是否运行中
        """
        if self._mcp_process and self._is_running:
            # 检查进程是否仍在运行
            return self._mcp_process.poll() is None
        return False

    def get_server_url(self):
        """
        获取 MCP 服务器 URL

        Returns:
            str: 服务器 URL
        """
        return f"http://localhost:{playwright_settings.port}/sse"


# 创建单例实例
playwright_mcp_manager = PlaywrightMCPManager()


# ========== 以下是 Playwright MCP 工具实现 ==========

class NavigateInput(BaseModel):
    """导航工具输入参数"""
    url: str = Field(..., description="要导航到的URL")

class PlaywrightNavigateTool(MCPTool):
    """Playwright导航工具"""
    name: str = "browser_navigate"
    description: str = "浏览器导航到指定URL"
    input_schema: type = NavigateInput

    async def _aexecute(
        self,
        url: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        执行导航操作

        Args:
            url: 要导航到的URL

        Returns:
            操作结果
        """
        # 这里应该调用实际的 Playwright MCP API
        # 在实际实现中，我们需要通过HTTP或其他方式与MCP服务通信
        # 这里仅作为示例
        return f"导航到 {url} 成功"


class SnapshotInput(BaseModel):
    """空输入参数"""
    pass


class PlaywrightSnapshotTool(MCPTool):
    """Playwright页面快照工具"""
    name: str = "browser_snapshot"
    description: str = "获取当前页面的可访问性快照，用于后续元素交互"
    input_schema: type = SnapshotInput

    async def _aexecute(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> Dict[str, Any]:
        """
        获取页面快照

        Returns:
            页面快照数据
        """
        # 实际实现中需要调用 Playwright MCP API
        return {"snapshot": "页面快照数据将在这里返回"}


class ClickInput(BaseModel):
    """点击操作输入参数"""
    element: str = Field(..., description="要点击的元素的人类可读描述")
    ref: str = Field(..., description="从页面快照中获取的元素精确引用")


class PlaywrightClickTool(MCPTool):
    """Playwright点击工具"""
    name: str = "browser_click"
    description: str = "在网页上执行点击操作"
    input_schema: type = ClickInput

    async def _aexecute(
        self,
        element: str,
        ref: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        执行点击操作

        Args:
            element: 元素描述
            ref: 元素引用

        Returns:
            操作结果
        """
        # 实际实现中需要调用 Playwright MCP API
        return f"点击元素 '{element}' 成功"


class TypeInput(BaseModel):
    """输入文本参数"""
    element: str = Field(..., description="要输入文本的元素的人类可读描述")
    ref: str = Field(..., description="从页面快照中获取的元素精确引用")
    text: str = Field(..., description="要输入的文本")
    submit: bool = Field(False, description="是否在输入后提交（按回车键）")
    slowly: bool = Field(False, description="是否逐字输入（触发按键处理程序）")


class PlaywrightTypeTool(MCPTool):
    """Playwright文本输入工具"""
    name: str = "browser_type"
    description: str = "在可编辑元素中输入文本"
    input_schema: type = TypeInput

    async def _aexecute(
        self,
        element: str,
        ref: str,
        text: str,
        submit: bool = False,
        slowly: bool = False,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        执行文本输入操作

        Args:
            element: 元素描述
            ref: 元素引用
            text: 输入文本
            submit: 是否提交
            slowly: 是否逐字输入

        Returns:
            操作结果
        """
        # 实际实现中需要调用 Playwright MCP API
        return f"在元素 '{element}' 中输入文本 '{text}' 成功"


class ScreenshotInput(BaseModel):
    """截图参数"""
    raw: bool = Field(False, description="是否返回未压缩的图像（PNG格式）")


class PlaywrightScreenshotTool(MCPTool):
    """Playwright截图工具"""
    name: str = "browser_take_screenshot"
    description: str = "获取当前页面的截图"
    input_schema: type = ScreenshotInput

    async def _aexecute(
        self,
        raw: bool = False,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        获取页面截图

        Args:
            raw: 是否返回未压缩图像

        Returns:
            截图数据（base64编码）
        """
        # 实际实现中需要调用 Playwright MCP API
        return "截图数据将在这里返回（base64编码）"


class WaitInput(BaseModel):
    """等待参数"""
    time: float = Field(..., description="等待的秒数（最多10秒）")


class PlaywrightWaitTool(MCPTool):
    """Playwright等待工具"""
    name: str = "browser_wait"
    description: str = "等待指定的时间（秒）"
    input_schema: type = WaitInput

    async def _aexecute(
        self,
        time: float,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        执行等待操作

        Args:
            time: 等待秒数

        Returns:
            操作结果
        """
        # 限制最大等待时间为10秒
        time = min(time, 10.0)
        await asyncio.sleep(time)
        return f"等待 {time} 秒完成"


def register_playwright_tools() -> None:
    """
    注册 Playwright MCP 工具到 MCP 工具管理器
    """
    if not playwright_settings.enabled:
        print("Playwright MCP 已禁用，跳过工具注册")
        return

    tools = [
        PlaywrightNavigateTool(),
        PlaywrightSnapshotTool(),
        PlaywrightClickTool(),
        PlaywrightTypeTool(),
        PlaywrightScreenshotTool(),
        PlaywrightWaitTool(),
    ]
    mcp_tool_manager.register_tools(tools)
    print(f"已注册 {len(tools)} 个 Playwright MCP 工具")


async def init_playwright_mcp() -> None:
    """
    初始化 Playwright MCP

    启动 MCP 服务器并注册工具
    """
    if not playwright_settings.enabled:
        print("Playwright MCP 已禁用，跳过初始化")
        return

    # 启动 MCP 服务器
    await playwright_mcp_manager.start_server()

    # 注册工具
    register_playwright_tools()

    print("Playwright MCP 工具已初始化")


def shutdown_playwright_mcp() -> None:
    """
    关闭 Playwright MCP
    """
    if not playwright_settings.enabled or not playwright_settings.auto_close:
        return

    playwright_mcp_manager.stop_server()
    print("Playwright MCP 已关闭")