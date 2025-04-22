"""
文档相关的数据模型
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class DocumentMetadata(BaseModel):
    """文档元数据模型"""
    source: Optional[str] = None
    page: Optional[int] = None
    file_path: Optional[str] = None
    file_type: Optional[str] = None
    file_name: Optional[str] = None
    creation_date: Optional[str] = None
    author: Optional[str] = None
    title: Optional[str] = None
    
    # 允许额外字段
    class Config:
        extra = "allow"

class DocumentContent(BaseModel):
    """文档内容模型"""
    page_content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DocumentResponse(BaseModel):
    """文档处理响应模型"""
    filename: str
    document_count: int
    documents: List[DocumentContent]

class SupportedFormatsResponse(BaseModel):
    """支持的文档格式响应模型"""
    formats: List[str]
