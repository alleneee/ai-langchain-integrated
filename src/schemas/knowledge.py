"""
知识库模式模块

该模块定义了知识库相关的请求和响应模式。
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from src.schemas.base import UserBase, PaginationParams

class CreateDatasetRequest(BaseModel):
    """创建数据集请求"""
    name: str = Field(..., description="数据集名称")

class DocumentListRequest(PaginationParams):
    """文档列表请求"""
    keyword: Optional[str] = Field(None, description="搜索关键词")

class SegmentModel(BaseModel):
    """文档片段模型"""
    content: str = Field(..., description="内容")
    answer: Optional[str] = Field(None, description="答案")
    keywords: Optional[List[str]] = Field(None, description="关键词列表")

class AddSegmentsRequest(BaseModel):
    """添加片段请求"""
    document_id: str = Field(..., description="文档ID")
    segments: List[SegmentModel] = Field(..., description="片段列表")

class QuerySegmentsRequest(BaseModel):
    """查询片段请求"""
    document_id: str = Field(..., description="文档ID")
    keyword: Optional[str] = Field(None, description="关键词")
    status: Optional[str] = Field(None, description="状态")

class UpdateSegmentRequest(BaseModel):
    """更新片段请求"""
    document_id: str = Field(..., description="文档ID")
    segment_id: str = Field(..., description="片段ID")
    segment_data: Dict[str, Any] = Field(..., description="片段数据")

class DeleteSegmentRequest(BaseModel):
    """删除片段请求"""
    document_id: str = Field(..., description="文档ID")
    segment_id: str = Field(..., description="片段ID")

# 响应模型
class Dataset(BaseModel):
    """数据集模型"""
    id: str = Field(..., description="数据集ID")
    name: str = Field(..., description="数据集名称")
    created_at: str = Field(..., description="创建时间")

class Document(BaseModel):
    """文档模型"""
    id: str = Field(..., description="文档ID")
    dataset_id: str = Field(..., description="数据集ID")
    name: str = Field(..., description="文档名称")
    status: str = Field(..., description="状态")
    created_at: str = Field(..., description="创建时间")

class Segment(BaseModel):
    """片段模型"""
    id: str = Field(..., description="片段ID")
    document_id: str = Field(..., description="文档ID")
    content: str = Field(..., description="内容")
    answer: Optional[str] = Field(None, description="答案")
    keywords: Optional[List[str]] = Field(None, description="关键词列表")
    created_at: str = Field(..., description="创建时间")
    updated_at: Optional[str] = Field(None, description="更新时间") 