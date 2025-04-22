"""
应用入口模块

该模块初始化并启动FastAPI应用。
"""

import os
# 设置 USER_AGENT 环境变量以避免警告
os.environ["USER_AGENT"] = "Dify-Connect/0.1.0"

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config.settings import settings
from src.api.router import api_router
from src.middlewares import register_exception_handlers

# 导入服务实例
from src.api.endpoints.llm import llm_service
# 导入新的 MCP 工具相关类
from src.utils.langchain_mcp_client import BaseMCPClient
from src.factories.mcp_tool_factory import MCPToolFactory
from src.utils.mcp_tools import PlaywrightMCPTool, BraveSearchMCPTool
# 导入异常类型
from langchain_core.tools import ToolException
# 导入配置
from src.config.playwright_config import playwright_settings
from src.config.brave_search_config import brave_search_settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 应用程序启动时的逻辑
    print("应用程序启动...")

    # 创建 MCP 客户端
    mcp_client = BaseMCPClient()
    app.state.mcp_client = mcp_client
    print("MCP 客户端初始化完成")

    # 初始化工具实例
    tools = []

    # 初始化 Playwright MCP 工具
    if playwright_settings.enabled:
        try:
            playwright_tool = PlaywrightMCPTool()
            await playwright_tool.initialize()
            app.state.playwright_tool = playwright_tool
            tools.append(playwright_tool)
            print("Playwright MCP 初始化完成")
        except Exception as e:
            print(f"Playwright MCP 初始化失败: {str(e)}")

    # 初始化 Brave Search MCP 工具
    if brave_search_settings.enabled:
        try:
            brave_search_tool = BraveSearchMCPTool()
            await brave_search_tool.initialize()
            app.state.brave_search_tool = brave_search_tool
            tools.append(brave_search_tool)
            print("Brave Search MCP 初始化完成")
        except Exception as e:
            print(f"Brave Search MCP 初始化失败: {str(e)}")

    # 注册工具到 MCP 服务
    if tools:
        try:
            # 添加to_dict方法如果不存在
            for tool in tools:
                if not hasattr(tool, 'to_dict'):
                    # 为工具添加一个简单的to_dict方法
                    tool.to_dict = lambda self=tool: {
                        "name": getattr(self, 'name', ''),
                        "description": getattr(self, 'description', ''),
                        "args_schema": getattr(self, 'args_schema', {})
                    }

            tool_dicts = [tool.to_dict() for tool in tools]
            await mcp_client.register_tools(tool_dicts)
            print(f"已注册 {len(tools)} 个工具到 MCP 服务")
        except Exception as e:
            print(f"工具注册失败: {str(e)}")

    # 初始化 LLM 服务
    await llm_service.initialize()
    print("LLM 服务初始化完成")

    # 可以在这里添加其他服务的初始化逻辑
    yield

    # 应用程序关闭时的逻辑
    print("应用程序关闭...")

    # 关闭 Playwright MCP
    if playwright_settings.enabled and playwright_settings.auto_close and hasattr(app.state, "playwright_tool"):
        await app.state.playwright_tool.shutdown()
        print("Playwright MCP 已关闭")

    # 关闭 Brave Search MCP
    if brave_search_settings.enabled and brave_search_settings.auto_close and hasattr(app.state, "brave_search_tool"):
        await app.state.brave_search_tool.shutdown()
        print("Brave Search MCP 已关闭")

def create_application() -> FastAPI:
    """创建FastAPI应用程序实例

    返回:
        FastAPI: 配置好的FastAPI应用程序实例
    """
    application = FastAPI(
        title=settings.APP_NAME,
        description=settings.APP_DESCRIPTION,
        version=settings.APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan  # 添加lifespan处理器
    )

    # 添加CORS中间件
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=settings.CORS_ALLOW_METHODS,
        allow_headers=settings.CORS_ALLOW_HEADERS,
    )

    # 注册异常处理器
    register_exception_handlers(application)

    # 注册API路由
    application.include_router(api_router, prefix=settings.API_PREFIX)

    # 添加根路由和健康检查路由
    @application.get("/")
    async def root():
        """根路径处理程序，返回应用基本信息"""
        return {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "description": settings.APP_DESCRIPTION,
        }

    @application.get("/health")
    async def health_check():
        """健康检查端点"""
        return {"status": "ok"}

    return application

app = create_application()

if __name__ == "__main__":
    # 启动Uvicorn服务器
    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,  # 仅在调试模式开启自动重载
        log_level="debug" if settings.DEBUG else "info",
    )
