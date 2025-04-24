"""
应用入口模块

该模块初始化并启动FastAPI应用。
"""

import os
# 设置 USER_AGENT 环境变量以避免警告
os.environ["USER_AGENT"] = "Dify-Connect/0.1.0"

import uvicorn
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
import time

from src.config.settings import settings
from src.api.router import api_router
from src.middlewares import register_exception_handlers
from src.db.mongodb import MongoDB

# 导入连接池管理器
from src.core.connection_pool import ConnectionPoolManager

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
    """应用程序生命周期管理器"""
    # 应用程序启动时的逻辑
    print("应用程序启动...")

    # 初始化连接池管理器
    try:
        pool_manager = ConnectionPoolManager.get_instance()
        await pool_manager.initialize(settings)
        app.state.pool_manager = pool_manager
        print("连接池管理器初始化完成")
    except Exception as e:
        print(f"连接池管理器初始化失败: {str(e)}")
        raise

    # 初始化数据库索引
    try:
        await MongoDB.create_indexes()
        print("MongoDB索引初始化完成")
    except Exception as e:
        print(f"MongoDB索引初始化失败: {str(e)}")

    # 创建 MCP 客户端
    try:
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
    except Exception as e:
        print(f"MCP 初始化失败: {str(e)}")

    # 初始化 LLM 服务
    try:
        await llm_service.initialize()
        print("LLM 服务初始化完成")
    except Exception as e:
        print(f"LLM 服务初始化失败: {str(e)}")

    # 可以在这里添加其他服务的初始化逻辑
    yield

    # 应用程序关闭时的逻辑
    print("应用程序关闭...")

    # 关闭 LLM 服务
    try:
        if hasattr(llm_service, 'provider_factory') and hasattr(llm_service.provider_factory, 'close'):
            await llm_service.provider_factory.close()
            print("LLM 服务已关闭")
    except Exception as e:
        print(f"关闭 LLM 服务时发生错误: {str(e)}")

    # 关闭 Playwright MCP
    if playwright_settings.enabled and playwright_settings.auto_close and hasattr(app.state, "playwright_tool"):
        await app.state.playwright_tool.shutdown()
        print("Playwright MCP 已关闭")

    # 关闭 Brave Search MCP
    if brave_search_settings.enabled and brave_search_settings.auto_close and hasattr(app.state, "brave_search_tool"):
        await app.state.brave_search_tool.shutdown()
        print("Brave Search MCP 已关闭")
        
    # 关闭连接池管理器
    try:
        await pool_manager.close()
        print("连接池管理器已关闭")
    except Exception as e:
        print(f"关闭连接池管理器时发生错误: {str(e)}")

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
    
    # 添加请求耗时中间件
    @application.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        """记录请求处理时间的中间件"""
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response

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
        """健康检查端点，检查各服务状态"""
        health_status = {
            "status": "ok",
            "services": {},
            "connections": {}
        }
        
        # 检查MongoDB连接
        try:
            db = await MongoDB.get_database()
            await db.command("ping")
            health_status["connections"]["mongodb"] = "ok"
        except Exception as e:
            health_status["connections"]["mongodb"] = {"status": "error", "message": str(e)}
            health_status["status"] = "degraded"
        
        # 检查LLM服务
        try:
            providers = await llm_service.get_supported_providers()
            health_status["services"]["llm"] = {
                "status": "ok",
                "providers": len([p for p in providers if p.get("available", False)])
            }
        except Exception as e:
            health_status["services"]["llm"] = {"status": "error", "message": str(e)}
            health_status["status"] = "degraded"
            
        return health_status

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
