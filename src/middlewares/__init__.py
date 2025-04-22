"""
中间件包

该包包含所有应用中间件。
"""

from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from src.middlewares.error_handler import (
    app_exception_handler,
    http_exception_handler,
    validation_exception_handler,
    general_exception_handler,
    llm_provider_exception_handler,
    rate_limit_exception_handler,
    database_exception_handler
)
from src.core.exceptions import (
    AppException, LLMProviderException, RateLimitException,
    DatabaseException
)

def register_exception_handlers(app: FastAPI):
    """
    注册异常处理程序
    
    注册所有异常处理中间件到FastAPI应用
    """
    # 注册自定义应用异常处理器
    app.add_exception_handler(AppException, app_exception_handler)
    
    # 注册特定类型异常处理器
    app.add_exception_handler(LLMProviderException, llm_provider_exception_handler)
    app.add_exception_handler(RateLimitException, rate_limit_exception_handler)
    app.add_exception_handler(DatabaseException, database_exception_handler)
    
    # 注册框架异常处理器
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    
    # 注册通用异常处理器（应最后注册，作为兜底）
    app.add_exception_handler(Exception, general_exception_handler)
