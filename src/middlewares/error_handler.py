"""
错误处理中间件模块

该模块提供了统一的异常处理中间件。
"""

import logging
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.core.exceptions import (
    AppException, LLMProviderException, LLMProviderAuthException,
    LLMProviderQuotaException, LLMProviderRateLimitException,
    LLMProviderModelNotFoundException, ValidationException,
    ConfigurationException, DatabaseException, ServiceInitException,
    FactoryException, AuthenticationException, RateLimitException,
    ThirdPartyServiceException, ResourceExhaustedException
)

logger = logging.getLogger(__name__)

async def app_exception_handler(request: Request, exc: AppException):
    """
    应用异常处理程序
    
    处理自定义应用异常
    """
    # 记录错误日志
    logger.error(f"应用异常: {exc.detail}", exc_info=True)
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "data": None,
            "error_code": getattr(exc, "error_code", None)
        },
    )

async def llm_provider_exception_handler(request: Request, exc: LLMProviderException):
    """
    LLM提供商异常处理程序
    
    处理所有LLM提供商相关的异常
    """
    # 记录错误日志
    logger.error(f"LLM提供商异常: {exc.detail}", exc_info=True)
    
    # 构建更详细的错误信息
    error_context = {}
    if hasattr(exc, "provider"):
        error_context["provider"] = exc.provider
    if hasattr(exc, "model"):
        error_context["model"] = exc.model
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "data": error_context if error_context else None,
            "error_code": getattr(exc, "error_code", "LLM_ERROR")
        },
    )

async def rate_limit_exception_handler(request: Request, exc: RateLimitException):
    """
    速率限制异常处理程序
    
    处理API速率限制相关的异常
    """
    # 记录警告日志
    logger.warning(f"速率限制异常: {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "data": None,
            "error_code": "RATE_LIMIT_EXCEEDED"
        },
    )

async def database_exception_handler(request: Request, exc: DatabaseException):
    """
    数据库异常处理程序
    
    处理数据库操作相关的异常
    """
    # 记录错误日志
    logger.error(f"数据库异常: {exc.detail}", exc_info=True)
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "data": None,
            "error_code": "DATABASE_ERROR"
        },
    )

async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """
    HTTP异常处理程序
    
    处理FastAPI和Starlette的HTTP异常
    """
    # 记录警告日志
    logger.warning(f"HTTP异常: {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": str(exc.detail),
            "data": None,
            "error_code": f"HTTP_{exc.status_code}"
        },
    )

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    请求验证异常处理程序
    
    处理请求参数验证错误
    """
    # 记录警告日志
    logger.warning(f"验证异常: {exc.errors()}")
    
    # 格式化验证错误信息，使其更易于理解
    formatted_errors = []
    for error in exc.errors():
        loc = error.get("loc", [])
        loc_str = " > ".join(str(item) for item in loc)
        msg = error.get("msg", "")
        formatted_errors.append(f"{loc_str}: {msg}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "message": "请求参数验证失败",
            "data": {
                "errors": formatted_errors,
                "details": exc.errors(),
            },
            "error_code": "VALIDATION_ERROR"
        },
    )

async def general_exception_handler(request: Request, exc: Exception):
    """
    通用异常处理程序
    
    处理未被其他处理程序捕获的异常
    """
    # 记录错误日志，包含堆栈跟踪
    logger.error(f"未捕获异常: {str(exc)}", exc_info=True)
    
    # 获取请求信息用于调试
    request_info = {
        "method": request.method,
        "url": str(request.url),
        "client": request.client.host if request.client else None,
        "headers": dict(request.headers),
    }
    
    # 在生产环境中不应返回详细错误信息
    from src.config.settings import settings
    if settings.APP_ENV == "production":
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "message": "服务器内部错误",
                "data": None,
                "error_code": "INTERNAL_SERVER_ERROR"
            },
        )
    else:
        # 开发环境下返回更详细的错误信息
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "message": f"服务器内部错误: {str(exc)}",
                "data": {
                    "exception_type": exc.__class__.__name__,
                    "request": request_info
                },
                "error_code": "INTERNAL_SERVER_ERROR"
            },
        )