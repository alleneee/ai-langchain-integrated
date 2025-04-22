"""
基础模式模块

该模块定义了API请求的基础模式。
"""

from pydantic import BaseModel, Field
from typing import Optional

class UserBase(BaseModel):
    """用户标识模型
    
    所有需要用户身份的请求都应继承此基类
    """
    user: str = Field(..., description="用户唯一标识")

class PaginationParams(BaseModel):
    """分页参数模型"""
    page: Optional[int] = Field(1, description="页码")
    page_size: Optional[int] = Field(10, description="每页记录数", le=100)
    
class OrderingParams(BaseModel):
    """排序参数模型"""
    order_by: Optional[str] = Field(None, description="排序字段")
    order_direction: Optional[str] = Field("asc", description="排序方向，asc或desc") 