"""LLM 通用异常处理工具"""

import logging
import json
from typing import Any, Dict

logger = logging.getLogger(__name__)

class LLMProviderError(Exception):
    """LLM 提供商基础异常类"""
    pass

class LLMAPIError(LLMProviderError):
    """API 调用相关错误"""
    def __init__(self, message: str, status_code: int = None, response_data: Dict[str, Any] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data

class LLMAuthenticationError(LLMAPIError):
    """认证相关错误"""
    pass

class LLMRateLimitError(LLMAPIError):
    """速率限制错误"""
    pass

class LLMQuotaError(LLMAPIError):
    """配额错误"""
    pass

class LLMModelNotAvailableError(LLMProviderError):
    """模型不可用错误"""
    pass

def format_exception(e: Exception, provider: str) -> str:
    """格式化 LLM 提供商的异常信息。"""
    error_type = type(e).__name__
    error_message = str(e)
    
    formatted_message = f"Error in {provider} provider ({error_type}): {error_message}"
    
    if isinstance(e, LLMAPIError):
        if e.status_code:
            formatted_message += f" (Status Code: {e.status_code})"
        if e.response_data:
            formatted_message += f" - Response: {e.response_data}"
            
    logger.error(formatted_message, exc_info=True)
    return formatted_message

def handle_llm_exception(exc: Exception, provider: str):
    """
    处理并记录 LLM 提供商的异常，然后抛出适当的自定义异常。

    Args:
        exc: 原始异常对象。
        provider: 产生异常的提供商名称（例如 'OpenAI', 'DeepSeek'）。

    Raises:
        LLMAuthenticationError: 如果是认证错误。
        LLMRateLimitError: 如果是速率限制错误。
        LLMQuotaError: 如果是配额错误。
        LLMModelNotAvailableError: 如果模型不可用或未找到。
        LLMAPIError: 其他与API相关的错误。
        LLMProviderError: 其他提供商级别的错误。
    """
    error_msg = str(exc)
    error_type_name = type(exc).__name__
    status_code = getattr(exc, 'status_code', getattr(exc, 'code', None)) # Try common attributes for status code
    response_data = getattr(exc, 'response', getattr(exc, 'body', None)) # Try common attributes for response data

    # Attempt to parse response if it looks like JSON
    parsed_response = None
    if isinstance(response_data, dict):
        parsed_response = response_data
    elif isinstance(response_data, str):
        try:
            parsed_response = json.loads(response_data)
        except json.JSONDecodeError:
            parsed_response = {"raw_content": response_data}
    elif hasattr(response_data, 'text'): # Handle requests.Response or similar
        try:
            parsed_response = response_data.json()
        except Exception:
            parsed_response = {"raw_content": getattr(response_data, 'text', str(response_data))}
    elif response_data is not None:
         parsed_response = {"raw_content": str(response_data)}

    # 先记录原始错误的基本信息
    log_message = f"Error in {provider} provider ({error_type_name}): {error_msg}"
    if status_code:
        log_message += f" Status Code: {status_code}"
    # 避免记录过多响应数据到日志
    log_message += f" Raw Exception Type: {type(exc)}"
    logger.error(log_message, exc_info=True) # Log with traceback

    # --- 异常分类逻辑 --- 
    lower_error_msg = error_msg.lower()
    
    # 认证错误
    auth_keywords = ["authentication", "auth", "key", "unauthorized", "permission", "权限", "认证", "api key", "invalid api key"]
    if any(keyword in lower_error_msg for keyword in auth_keywords) or status_code == 401:
         detail = f"{provider} 认证失败: {error_msg}"
         raise LLMAuthenticationError(detail, status_code=status_code, response_data=parsed_response) from exc

    # 配额/账单错误
    quota_keywords = ["quota", "billing", "payment", "insufficient_quota", "支付", "账单", "配额"]
    # Check for 429 specifically related to quota
    is_quota_related_429 = (status_code == 429 and any(keyword in lower_error_msg for keyword in quota_keywords))
    if any(keyword in lower_error_msg for keyword in quota_keywords) or is_quota_related_429:
        detail = f"{provider} 配额或计费问题: {error_msg}"
        raise LLMQuotaError(detail, status_code=status_code, response_data=parsed_response) from exc

    # 速率限制错误 (Catch 429 if not specifically quota related)
    rate_limit_keywords = ["rate limit", "too many requests", "ratelimit", "速率", "频率"]
    if any(keyword in lower_error_msg for keyword in rate_limit_keywords) or (status_code == 429 and not is_quota_related_429):
        detail = f"{provider} 请求频率过高: {error_msg}"
        raise LLMRateLimitError(detail, status_code=status_code, response_data=parsed_response) from exc

    # 模型不存在错误
    model_not_found_keywords = ["model not found", "model doesn't exist", "模型不存在", "not found"]
    # Gemini uses 400 for invalid model sometimes, check message
    is_model_not_found_msg = any(keyword in lower_error_msg for keyword in model_not_found_keywords)
    if is_model_not_found_msg or status_code == 404:
        detail = f"{provider} 模型未找到或不可用: {error_msg}"
        raise LLMModelNotAvailableError(detail) from exc

    # 其他 API 错误 (例如，4xx/5xx 状态码但不是上述特定类型)
    if status_code and status_code >= 400:
         detail = f"{provider} API 调用错误 (Status: {status_code}, Type: {error_type_name}): {error_msg}"
         raise LLMAPIError(detail, status_code=status_code, response_data=parsed_response) from exc

    # 其他通用提供商错误 (包括网络连接错误等)
    detail = f"{provider} 调用时发生错误 ({error_type_name}): {error_msg}"
    raise LLMProviderError(detail) from exc

# 你可以在这里添加更多异常处理函数或自定义异常
