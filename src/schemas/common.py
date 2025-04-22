"""
通用响应模式模块

定义系统中通用的数据模型和响应格式
"""

from typing import Generic, TypeVar, Optional, Any
from pydantic import BaseModel, Field

# 用于泛型响应的类型变量
T = TypeVar('T')

class BaseResponse(BaseModel, Generic[T]):
    """
    通用API响应基类

    用于包装各种API响应数据，提供统一的格式

    属性:
        success: 操作是否成功
        message: 操作结果消息
        data: 响应数据，类型由泛型参数T决定
    """
    success: bool = Field(True, description="操作是否成功")
    message: str = Field("操作成功", description="操作结果消息")
    data: Optional[T] = Field(None, description="响应数据")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": True,
                    "message": "操作成功",
                    "data": None
                }
            ]
        }
    }