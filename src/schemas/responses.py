"""
响应模式模块

该模块定义了API响应的标准格式。
"""

from typing import Optional, Any, Generic, TypeVar, List, Dict
from pydantic import BaseModel, Field

# 用于泛型响应的类型变量
T = TypeVar('T')

class StandardResponse(BaseModel):
    """标准API响应格式"""
    success: bool = Field(True, description="操作是否成功")
    message: str = Field("操作成功", description="操作消息")
    data: Optional[Any] = Field(None, description="响应数据")

class DataResponse(BaseModel, Generic[T]):
    """带泛型数据的API响应格式"""
    success: bool = Field(True, description="操作是否成功")
    message: str = Field("操作成功", description="操作消息")
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

class PaginatedResponse(BaseModel, Generic[T]):
    """分页响应格式"""
    success: bool = Field(True, description="操作是否成功")
    message: str = Field("操作成功", description="操作消息")
    data: List[T] = Field([], description="响应数据列表")
    total: int = Field(0, description="总记录数")
    page: int = Field(1, description="当前页码")
    page_size: int = Field(10, description="每页记录数")
    pages: int = Field(1, description="总页数")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": True,
                    "message": "操作成功",
                    "data": [],
                    "total": 0,
                    "page": 1,
                    "page_size": 10,
                    "pages": 1
                }
            ]
        }
    }