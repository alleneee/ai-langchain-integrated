"""
应用程序配置模块
"""

import os
from typing import List, Optional
from pydantic import BaseSettings

class Settings(BaseSettings):
    """应用程序配置设置"""
    # 基本应用配置
    APP_NAME: str = "旅游助手API"
    API_PREFIX: str = "/api"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # CORS配置
    CORS_ORIGINS: List[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    # LLM配置
    DEFAULT_MODEL: str = "gpt-3.5-turbo"
    DEFAULT_TEMPERATURE: float = 0.7
    
    # 高德地图MCP配置
    AMAP_API_KEY: Optional[str] = None
    
    # Brave Search MCP配置
    BRAVE_SEARCH_MCP_ENABLED: bool = True
    BRAVE_SEARCH_MCP_KEY: str = "cjtvdam9m9bc3q"
    BRAVE_SEARCH_MCP_AUTO_CLOSE: bool = True
    BRAVE_SEARCH_MCP_MAX_RESULTS: int = 10
    BRAVE_SEARCH_MCP_LANGUAGE: str = "zh-CN"
    BRAVE_SEARCH_MCP_REGION: str = "CN"
    
    # Playwright MCP配置
    PLAYWRIGHT_MCP_ENABLED: bool = True
    PLAYWRIGHT_MCP_PORT: int = 8089
    PLAYWRIGHT_MCP_AUTO_CLOSE: bool = True
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# 创建配置对象实例
settings = Settings()

# 从环境变量覆盖配置 
if os.getenv("APP_NAME"):
    settings.APP_NAME = os.getenv("APP_NAME")
    
if os.getenv("DEBUG"):
    settings.DEBUG = os.getenv("DEBUG").lower() == "true"
    
if os.getenv("DEFAULT_MODEL"):
    settings.DEFAULT_MODEL = os.getenv("DEFAULT_MODEL")
    
if os.getenv("DEFAULT_TEMPERATURE"):
    settings.DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE"))
    
# 高德地图配置
if os.getenv("AMAP_API_KEY"):
    settings.AMAP_API_KEY = os.getenv("AMAP_API_KEY")
    
# Brave Search MCP配置
if os.getenv("BRAVE_SEARCH_MCP_ENABLED"):
    settings.BRAVE_SEARCH_MCP_ENABLED = os.getenv("BRAVE_SEARCH_MCP_ENABLED").lower() == "true"
    
if os.getenv("BRAVE_SEARCH_MCP_KEY"):
    settings.BRAVE_SEARCH_MCP_KEY = os.getenv("BRAVE_SEARCH_MCP_KEY")
    
if os.getenv("BRAVE_SEARCH_MCP_AUTO_CLOSE"):
    settings.BRAVE_SEARCH_MCP_AUTO_CLOSE = os.getenv("BRAVE_SEARCH_MCP_AUTO_CLOSE").lower() == "true"
    
if os.getenv("BRAVE_SEARCH_MCP_MAX_RESULTS"):
    settings.BRAVE_SEARCH_MCP_MAX_RESULTS = int(os.getenv("BRAVE_SEARCH_MCP_MAX_RESULTS"))

if os.getenv("BRAVE_SEARCH_MCP_LANGUAGE"):
    settings.BRAVE_SEARCH_MCP_LANGUAGE = os.getenv("BRAVE_SEARCH_MCP_LANGUAGE")

if os.getenv("BRAVE_SEARCH_MCP_REGION"):
    settings.BRAVE_SEARCH_MCP_REGION = os.getenv("BRAVE_SEARCH_MCP_REGION")
    
# Playwright MCP配置
if os.getenv("PLAYWRIGHT_MCP_ENABLED"):
    settings.PLAYWRIGHT_MCP_ENABLED = os.getenv("PLAYWRIGHT_MCP_ENABLED").lower() == "true"
    
if os.getenv("PLAYWRIGHT_MCP_PORT"):
    settings.PLAYWRIGHT_MCP_PORT = int(os.getenv("PLAYWRIGHT_MCP_PORT"))
    
if os.getenv("PLAYWRIGHT_MCP_AUTO_CLOSE"):
    settings.PLAYWRIGHT_MCP_AUTO_CLOSE = os.getenv("PLAYWRIGHT_MCP_AUTO_CLOSE").lower() == "true" 