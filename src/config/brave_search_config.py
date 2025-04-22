"""
Brave Search MCP 配置模块

管理 Brave Search MCP 的配置设置
"""

import os
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List


class BraveSearchMCPSettings(BaseModel):
    """
    Brave Search MCP 设置

    控制 Brave Search MCP 服务器和工具的行为
    """
    enabled: bool = Field(
        default=True,
        description="是否启用 Brave Search MCP"
    )

    server_key: str = Field(
        default="cjtvdam9m9bc3q",
        description="Brave Search MCP 服务器密钥"
    )

    api_key: str = Field(
        default="",
        description="Brave Search API 密钥"
    )

    api_base: str = Field(
        default="https://api.search.brave.com",
        description="Brave Search API 基础URL"
    )

    auto_close: bool = Field(
        default=True,
        description="应用程序关闭时是否自动关闭 MCP 服务器"
    )

    max_results: int = Field(
        default=10,
        description="搜索结果最大数量"
    )

    timeout: int = Field(
        default=30000,
        description="操作超时时间（毫秒）"
    )

    mcp_router_path: str = Field(
        default="mcprouter",
        description="MCP Router 包路径"
    )

    search_language: str = Field(
        default="zh-CN",
        description="搜索语言设置"
    )

    search_region: str = Field(
        default="CN",
        description="搜索区域设置"
    )

    def get_server_config(self) -> Dict[str, Any]:
        """
        获取服务器配置

        Returns:
            Dict[str, Any]: MCP服务器配置字典
        """
        return {
            "mcpServers": {
                "": {
                    "command": "npx",
                    "args": [
                        "-y",
                        self.mcp_router_path
                    ],
                    "env": {
                        "SERVER_KEY": self.server_key
                    }
                }
            }
        }


# 默认配置
brave_search_settings = BraveSearchMCPSettings()

# 从环境变量加载配置
if os.environ.get("BRAVE_SEARCH_MCP_ENABLED"):
    brave_search_settings.enabled = os.environ.get("BRAVE_SEARCH_MCP_ENABLED").lower() == "true"

if os.environ.get("BRAVE_SEARCH_MCP_SERVER_KEY"):
    brave_search_settings.server_key = os.environ.get("BRAVE_SEARCH_MCP_SERVER_KEY")

if os.environ.get("BRAVE_SEARCH_API_KEY"):
    brave_search_settings.api_key = os.environ.get("BRAVE_SEARCH_API_KEY")

if os.environ.get("BRAVE_SEARCH_API_BASE"):
    brave_search_settings.api_base = os.environ.get("BRAVE_SEARCH_API_BASE")

if os.environ.get("BRAVE_SEARCH_MCP_AUTO_CLOSE"):
    brave_search_settings.auto_close = os.environ.get("BRAVE_SEARCH_MCP_AUTO_CLOSE").lower() == "true"

if os.environ.get("BRAVE_SEARCH_MCP_MAX_RESULTS"):
    brave_search_settings.max_results = int(os.environ.get("BRAVE_SEARCH_MCP_MAX_RESULTS"))

if os.environ.get("BRAVE_SEARCH_MCP_TIMEOUT"):
    brave_search_settings.timeout = int(os.environ.get("BRAVE_SEARCH_MCP_TIMEOUT"))

if os.environ.get("BRAVE_SEARCH_MCP_ROUTER_PATH"):
    brave_search_settings.mcp_router_path = os.environ.get("BRAVE_SEARCH_MCP_ROUTER_PATH")

if os.environ.get("BRAVE_SEARCH_MCP_LANGUAGE"):
    brave_search_settings.search_language = os.environ.get("BRAVE_SEARCH_MCP_LANGUAGE")

if os.environ.get("BRAVE_SEARCH_MCP_REGION"):
    brave_search_settings.search_region = os.environ.get("BRAVE_SEARCH_MCP_REGION")