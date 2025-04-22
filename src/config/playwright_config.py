"""
Playwright MCP 配置模块

管理 Playwright MCP 的配置设置
"""

import os
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional


class PlaywrightMCPSettings(BaseModel):
    """
    Playwright MCP 设置
    
    控制 Playwright MCP 服务器和工具的行为
    """
    enabled: bool = Field(
        default=True,
        description="是否启用 Playwright MCP"
    )
    
    headless: bool = Field(
        default=True,
        description="是否以无头模式运行浏览器"
    )
    
    port: int = Field(
        default=8080,
        description="MCP 服务器端口"
    )
    
    snapshot_mode: bool = Field(
        default=True,
        description="是否使用快照模式（如果为False则使用视觉模式）"
    )
    
    browser: str = Field(
        default="chromium",
        description="使用的浏览器类型 (chromium, firefox, webkit)"
    )
    
    timeout: int = Field(
        default=30000,
        description="操作超时时间（毫秒）"
    )
    
    viewport_width: int = Field(
        default=1280,
        description="浏览器视口宽度"
    )
    
    viewport_height: int = Field(
        default=720,
        description="浏览器视口高度"
    )
    
    auto_close: bool = Field(
        default=True,
        description="应用程序关闭时是否自动关闭 MCP 服务器"
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        return self.model_dump()


# 默认配置
playwright_settings = PlaywrightMCPSettings()

# 从环境变量加载配置
if os.environ.get("PLAYWRIGHT_MCP_ENABLED"):
    playwright_settings.enabled = os.environ.get("PLAYWRIGHT_MCP_ENABLED").lower() == "true"

if os.environ.get("PLAYWRIGHT_MCP_HEADLESS"):
    playwright_settings.headless = os.environ.get("PLAYWRIGHT_MCP_HEADLESS").lower() == "true"

if os.environ.get("PLAYWRIGHT_MCP_PORT"):
    playwright_settings.port = int(os.environ.get("PLAYWRIGHT_MCP_PORT"))

if os.environ.get("PLAYWRIGHT_MCP_SNAPSHOT_MODE"):
    playwright_settings.snapshot_mode = os.environ.get("PLAYWRIGHT_MCP_SNAPSHOT_MODE").lower() == "true"

if os.environ.get("PLAYWRIGHT_MCP_BROWSER"):
    playwright_settings.browser = os.environ.get("PLAYWRIGHT_MCP_BROWSER")

if os.environ.get("PLAYWRIGHT_MCP_TIMEOUT"):
    playwright_settings.timeout = int(os.environ.get("PLAYWRIGHT_MCP_TIMEOUT"))

if os.environ.get("PLAYWRIGHT_MCP_VIEWPORT_WIDTH"):
    playwright_settings.viewport_width = int(os.environ.get("PLAYWRIGHT_MCP_VIEWPORT_WIDTH"))

if os.environ.get("PLAYWRIGHT_MCP_VIEWPORT_HEIGHT"):
    playwright_settings.viewport_height = int(os.environ.get("PLAYWRIGHT_MCP_VIEWPORT_HEIGHT"))

if os.environ.get("PLAYWRIGHT_MCP_AUTO_CLOSE"):
    playwright_settings.auto_close = os.environ.get("PLAYWRIGHT_MCP_AUTO_CLOSE").lower() == "true" 