"""
异常处理模块

该模块定义了应用中使用的自定义异常类。
"""

class AppException(Exception):
    """应用基础异常类"""
    
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(self.detail)

class NotFoundException(AppException):
    """资源未找到异常"""
    
    def __init__(self, detail: str = "请求的资源不存在"):
        super().__init__(status_code=404, detail=detail)

class BadRequestException(AppException):
    """请求参数错误异常"""
    
    def __init__(self, detail: str = "请求参数错误"):
        super().__init__(status_code=400, detail=detail)

class UnauthorizedException(AppException):
    """未授权异常"""
    
    def __init__(self, detail: str = "未经授权的请求"):
        super().__init__(status_code=401, detail=detail)

class ForbiddenException(AppException):
    """禁止访问异常"""
    
    def __init__(self, detail: str = "禁止访问该资源"):
        super().__init__(status_code=403, detail=detail)

class ServerException(AppException):
    """服务器内部错误异常"""
    
    def __init__(self, detail: str = "服务器内部错误"):
        super().__init__(status_code=500, detail=detail)

class LLMProviderException(AppException):
    """LLM提供商错误异常"""
    
    def __init__(self, detail: str = "LLM提供商调用失败"):
        super().__init__(status_code=500, detail=detail)

# 添加新的异常类

class ValidationException(BadRequestException):
    """数据验证异常"""
    
    def __init__(self, detail: str = "数据验证失败"):
        super().__init__(detail=detail)

class ConfigurationException(ServerException):
    """配置错误异常"""
    
    def __init__(self, detail: str = "系统配置错误"):
        super().__init__(detail=detail)

class DatabaseException(ServerException):
    """数据库操作异常"""
    
    def __init__(self, detail: str = "数据库操作失败"):
        super().__init__(detail=detail)

class ServiceInitException(ServerException):
    """服务初始化异常"""
    
    def __init__(self, detail: str = "服务初始化失败"):
        super().__init__(detail=detail)

class FactoryException(ServerException):
    """工厂创建对象异常"""
    
    def __init__(self, detail: str = "创建对象失败"):
        super().__init__(detail=detail)

class AuthenticationException(UnauthorizedException):
    """认证失败异常"""
    
    def __init__(self, detail: str = "用户认证失败"):
        super().__init__(detail=detail)

class RateLimitException(AppException):
    """速率限制异常"""
    
    def __init__(self, detail: str = "请求过于频繁，请稍后再试"):
        super().__init__(status_code=429, detail=detail)

class ThirdPartyServiceException(ServerException):
    """第三方服务调用异常"""
    
    def __init__(self, detail: str = "第三方服务调用失败"):
        super().__init__(detail=detail)

class ResourceExhaustedException(AppException):
    """资源耗尽异常"""
    
    def __init__(self, detail: str = "系统资源已耗尽"):
        super().__init__(status_code=503, detail=detail)

class LLMProviderAuthException(LLMProviderException):
    """LLM提供商认证异常"""
    
    def __init__(self, detail: str = "LLM提供商认证失败"):
        super().__init__(detail=detail)

class LLMProviderQuotaException(LLMProviderException):
    """LLM提供商配额异常"""
    
    def __init__(self, detail: str = "LLM提供商配额已用尽"):
        super().__init__(detail=detail)

class LLMProviderRateLimitException(LLMProviderException):
    """LLM提供商速率限制异常"""
    
    def __init__(self, detail: str = "LLM提供商速率限制，请稍后再试"):
        super().__init__(detail=detail)

class LLMProviderModelNotFoundException(LLMProviderException):
    """LLM提供商模型未找到异常"""
    
    def __init__(self, detail: str = "请求的LLM模型不存在"):
        super().__init__(detail=detail)