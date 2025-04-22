"""
通用依赖模块

该模块提供应用中使用的通用依赖函数。
"""

from src.config.settings import get_settings

def get_settings_dependency():
    """设置依赖函数
    
    用于依赖注入，提供配置实例。
    """
    return get_settings() 