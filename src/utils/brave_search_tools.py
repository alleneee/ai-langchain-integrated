"""
Brave Search MCP 工具实现模块

本模块提供了基于 Brave Search MCP 的搜索工具实现
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

# 导入 Brave Search 配置
from src.config.brave_search_config import brave_search_settings


class BraveSearchMCPManager:
    """
    Brave Search MCP 管理器

    用于启动、停止 Brave Search MCP 服务器并管理与其的通信
    """
    _instance = None
    _initialized = False
    _mcp_process = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BraveSearchMCPManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not BraveSearchMCPManager._initialized:
            self._is_running = False
            BraveSearchMCPManager._initialized = True

    async def start_server(self):
        """
        启动 Brave Search MCP 服务器
        """
        if not brave_search_settings.enabled:
            print("Brave Search MCP 已禁用，跳过服务器启动")
            return

        if self._is_running:
            return

        # 获取服务器配置
        config = brave_search_settings.get_server_config()

        # 构建命令行参数
        args = [config["mcpServers"][""]["command"]]
        args.extend(config["mcpServers"][""]["args"])

        # 设置环境变量
        env = os.environ.copy()
        for key, value in config["mcpServers"][""]["env"].items():
            env[key] = value

        # 启动进程
        try:
            self._mcp_process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )
            print(f"Brave Search MCP 服务器已启动")

            # 等待服务器启动
            await asyncio.sleep(2)
            self._is_running = True
        except Exception as e:
            print(f"启动 Brave Search MCP 服务器失败: {str(e)}")
            raise

    def stop_server(self):
        """
        停止 Brave Search MCP 服务器
        """
        if not self._is_running or self._mcp_process is None:
            return

        try:
            self._mcp_process.terminate()
            self._mcp_process.wait(timeout=5)
            self._is_running = False
            print("Brave Search MCP 服务器已停止")
        except Exception as e:
            print(f"停止 Brave Search MCP 服务器失败: {str(e)}")
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


# 创建单例实例
brave_search_manager = BraveSearchMCPManager()


# ========== 以下是 Brave Search MCP 工具实现 ==========

class WebSearchInput(BaseModel):
    """网络搜索工具输入参数"""
    search_term: str = Field(..., description="要搜索的关键词")
    explanation: Optional[str] = Field(None, description="搜索原因说明")
    max_results: Optional[int] = Field(None, description="返回结果的最大数量")


class SearchResult(BaseModel):
    """搜索结果项"""
    title: str = Field(..., description="结果标题")
    url: str = Field(..., description="结果URL")
    description: str = Field(..., description="结果描述")


class BraveWebSearchTool(MCPTool if MCP_AVAILABLE else object):
    """Brave Web搜索工具"""
    name: str = "brave_web_search"
    description: str = "使用Brave搜索引擎搜索网络以获取实时信息"
    input_schema: type = WebSearchInput

    async def _aexecute(
        self,
        search_term: str,
        explanation: Optional[str] = None,
        max_results: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List[Dict[str, Any]]:
        """
        执行网络搜索

        Args:
            search_term: 搜索关键词
            explanation: 搜索原因说明（可选）
            max_results: 返回结果的最大数量（可选）

        Returns:
            搜索结果列表
        """
        # 确保服务器正在运行
        if not brave_search_manager.is_running:
            if brave_search_settings.enabled:
                await brave_search_manager.start_server()
            else:
                return [{"error": "Brave Search MCP 已禁用"}]

        # 确定结果数量
        result_count = max_results if max_results is not None else brave_search_settings.max_results

        # 模拟搜索结果（实际实现中需要调用真实API）
        # 在实际实现中，需要通过HTTP或其他方式与MCP服务通信
        mock_results = []
        for i in range(min(3, result_count)):  # 模拟最多3个结果
            mock_results.append({
                "title": f"搜索结果 {i+1} 的标题 - {search_term}",
                "url": f"https://example.com/result{i+1}?q={search_term}",
                "description": f"这是关于'{search_term}'的搜索结果{i+1}的描述文本。在实际实现中，这里会包含真实的搜索结果摘要。"
            })

        return mock_results


class BraveNewsSearchTool(MCPTool if MCP_AVAILABLE else object):
    """Brave新闻搜索工具"""
    name: str = "brave_news_search"
    description: str = "使用Brave搜索引擎搜索最新新闻"
    input_schema: type = WebSearchInput

    async def _aexecute(
        self,
        search_term: str,
        explanation: Optional[str] = None,
        max_results: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List[Dict[str, Any]]:
        """
        执行新闻搜索

        Args:
            search_term: 搜索关键词
            explanation: 搜索原因说明（可选）
            max_results: 返回结果的最大数量（可选）

        Returns:
            新闻搜索结果列表
        """
        # 确保服务器正在运行
        if not brave_search_manager.is_running:
            if brave_search_settings.enabled:
                await brave_search_manager.start_server()
            else:
                return [{"error": "Brave Search MCP 已禁用"}]

        # 确定结果数量
        result_count = max_results if max_results is not None else brave_search_settings.max_results

        # 模拟新闻搜索结果（实际实现中需要调用真实API）
        mock_results = []
        for i in range(min(3, result_count)):  # 模拟最多3个结果
            mock_results.append({
                "title": f"关于'{search_term}'的最新新闻 {i+1}",
                "url": f"https://news.example.com/article{i+1}?topic={search_term}",
                "description": f"这是关于'{search_term}'的最新新闻文章{i+1}。发布于模拟日期。在实际实现中，这里会包含真实的新闻摘要和日期。",
                "source": f"新闻来源 {i+1}",
                "published_date": "2023-06-01"  # 模拟日期
            })

        return mock_results


class BraveImageSearchTool(MCPTool if MCP_AVAILABLE else object):
    """Brave图片搜索工具"""
    name: str = "brave_image_search"
    description: str = "使用Brave搜索引擎搜索图片"
    input_schema: type = WebSearchInput

    async def _aexecute(
        self,
        search_term: str,
        explanation: Optional[str] = None,
        max_results: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List[Dict[str, Any]]:
        """
        执行图片搜索

        Args:
            search_term: 搜索关键词
            explanation: 搜索原因说明（可选）
            max_results: 返回结果的最大数量（可选）

        Returns:
            图片搜索结果列表
        """
        # 确保服务器正在运行
        if not brave_search_manager.is_running:
            if brave_search_settings.enabled:
                await brave_search_manager.start_server()
            else:
                return [{"error": "Brave Search MCP 已禁用"}]

        # 确定结果数量
        result_count = max_results if max_results is not None else brave_search_settings.max_results

        # 模拟图片搜索结果（实际实现中需要调用真实API）
        mock_results = []
        for i in range(min(3, result_count)):  # 模拟最多3个结果
            mock_results.append({
                "title": f"关于'{search_term}'的图片 {i+1}",
                "url": f"https://images.example.com/image{i+1}?q={search_term}",
                "image_url": f"https://images.example.com/image{i+1}.jpg",
                "source_website": f"https://source{i+1}.example.com/",
                "width": 800,
                "height": 600
            })

        return mock_results


def register_brave_search_tools() -> None:
    """
    注册 Brave Search MCP 工具到 MCP 工具管理器
    """
    if not brave_search_settings.enabled:
        print("Brave Search MCP 已禁用，跳过工具注册")
        return

    tools = [
        BraveWebSearchTool(),
        BraveNewsSearchTool(),
        BraveImageSearchTool()
    ]
    mcp_tool_manager.register_tools(tools)
    print(f"已注册 {len(tools)} 个 Brave Search MCP 工具")


async def init_brave_search_mcp() -> None:
    """
    初始化 Brave Search MCP

    启动 MCP 服务器并注册工具
    """
    if not brave_search_settings.enabled:
        print("Brave Search MCP 已禁用，跳过初始化")
        return

    # 启动 MCP 服务器
    await brave_search_manager.start_server()

    # 注册工具
    register_brave_search_tools()

    print("Brave Search MCP 工具已初始化")


def shutdown_brave_search_mcp() -> None:
    """
    关闭 Brave Search MCP
    """
    if not brave_search_settings.enabled or not brave_search_settings.auto_close:
        return

    brave_search_manager.stop_server()
    print("Brave Search MCP 已关闭")