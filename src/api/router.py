"""
API路由模块

该模块整合所有API路由。
"""

from fastapi import APIRouter

from src.api.endpoints import (
    chat,
    completion,
    workflow,
    knowledge,
    llm,
    chatbot
)
# 导入工具路由
from src.api.endpoints.tools import router as tools_router
# 导入Playwright演示路由
from src.api.endpoints.playwright_demo import router as playwright_router
# 导入Brave Search演示路由
from src.api.endpoints.brave_search_demo import router as brave_search_router
# 导入新的 MCP 演示路由
from src.api.endpoints.mcp_demo import router as mcp_demo_router
# 导入文档处理路由
from src.api.endpoints.documents import router as documents_router
# 导入异步文档处理路由
from src.api.endpoints.documents_async import router as documents_async_router
# 导入向量知识库路由
from src.api.endpoints.vector_knowledge import router as vector_knowledge_router
# 导入异步向量知识库路由
from src.api.endpoints.vector_knowledge_async import router as vector_knowledge_async_router

# 创建主API路由器
api_router = APIRouter()

# 注册各个功能模块的路由
api_router.include_router(chat.router, prefix="/chat", tags=["聊天"])
api_router.include_router(completion.router, prefix="/completion", tags=["补全"])
api_router.include_router(workflow.router, prefix="/workflow", tags=["工作流"])
api_router.include_router(knowledge.router, prefix="/knowledge", tags=["知识库"])
api_router.include_router(llm.router, prefix="/llm", tags=["LLM"])
api_router.include_router(chatbot.router, prefix="/chatbot", tags=["聊天机器人"])
# 添加工具路由
api_router.include_router(tools_router, prefix="/tools", tags=["工具"])
# 添加Playwright演示路由
api_router.include_router(playwright_router, prefix="/playwright", tags=["Playwright"])
# 添加Brave Search演示路由
api_router.include_router(brave_search_router, prefix="/brave-search", tags=["Brave Search"])
# 添加新的 MCP 演示路由
api_router.include_router(mcp_demo_router, prefix="/mcp", tags=["MCP 工具"])
# 添加文档处理路由
api_router.include_router(documents_router, prefix="/documents", tags=["文档处理"])
# 添加异步文档处理路由
api_router.include_router(documents_async_router, prefix="/documents/async", tags=["异步文档处理"])
# 添加向量知识库路由
api_router.include_router(vector_knowledge_router, prefix="/vector-kb", tags=["向量知识库"])
# 添加异步向量知识库路由
api_router.include_router(vector_knowledge_async_router, prefix="/vector-kb/async", tags=["异步向量知识库"])