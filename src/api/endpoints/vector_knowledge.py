"""
向量知识库API端点模块

提供本地向量存储的知识库相关API端点
"""

from fastapi import APIRouter, Depends, HTTPException, status, Path, Query, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import os
import logging
from pathlib import Path as FilePath

from src.config.settings import get_settings
from src.services.knowledge_base.kb_manager import get_kb_manager, KnowledgeBaseManager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter()

# --- 数据模型 ---

class KnowledgeBaseStatusResponse(BaseModel):
    """知识库状态响应模型"""
    message: str
    status: str
    vector_store_type: str
    vector_store_dir: str

class DocumentAddRequest(BaseModel):
    """添加文档请求模型"""
    source_path: str = Field(..., description="文档源路径，可以是文件或目录")
    collection_name: Optional[str] = Field(None, description="集合名称")

class DocumentAddResponse(BaseModel):
    """添加文档响应模型"""
    status: str
    message: str

class QueryRequest(BaseModel):
    """查询请求模型"""
    query_text: str = Field(..., description="查询文本")
    collection_name: Optional[str] = Field(None, description="集合名称")
    filter_metadata: Optional[Dict[str, Any]] = Field(None, description="元数据过滤")
    n_results: Optional[int] = Field(5, description="返回结果数量")

class DocumentContent(BaseModel):
    """文档内容模型"""
    content: str
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    """查询响应模型"""
    results: List[DocumentContent]
    query: str

class CollectionResponse(BaseModel):
    """集合响应模型"""
    collections: List[str]

class CollectionStatsResponse(BaseModel):
    """集合统计信息响应模型"""
    status: str
    message: Optional[str] = None
    stats: Optional[Dict[str, Any]] = None


# --- 依赖项 ---

def get_kb_manager_dependency():
    """获取知识库管理器依赖"""
    return get_kb_manager()


# --- API端点 ---

@router.get("/status", response_model=KnowledgeBaseStatusResponse)
async def get_knowledge_base_status(
    kb_manager: KnowledgeBaseManager = Depends(get_kb_manager_dependency)
):
    """
    获取向量知识库状态
    
    Returns:
        KnowledgeBaseStatusResponse: 知识库状态响应
    """
    settings = get_settings()
    return KnowledgeBaseStatusResponse(
        message="向量知识库功能已可用",
        status="available",
        vector_store_type=settings.VECTOR_STORE_TYPE,
        vector_store_dir=settings.VECTOR_STORE_DIR
    )

@router.post("/documents", response_model=DocumentAddResponse, status_code=status.HTTP_201_CREATED)
async def add_documents(
    request: DocumentAddRequest,
    kb_manager: KnowledgeBaseManager = Depends(get_kb_manager_dependency)
):
    """
    添加文档到知识库
    
    Args:
        request: 添加文档请求
        
    Returns:
        DocumentAddResponse: 添加结果
    """
    try:
        # 检查源路径是否存在
        if not os.path.exists(request.source_path):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"源路径不存在: {request.source_path}"
            )
            
        result = kb_manager.add_documents(
            source_path=request.source_path,
            collection_name=request.collection_name
        )
        
        return DocumentAddResponse(
            status=result["status"],
            message=result["message"]
        )
    except Exception as e:
        logger.error(f"添加文档失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"添加文档失败: {str(e)}"
        )

@router.post("/upload", response_model=DocumentAddResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    collection_name: Optional[str] = Form(None),
    kb_manager: KnowledgeBaseManager = Depends(get_kb_manager_dependency)
):
    """
    上传文档到知识库
    
    Args:
        file: 上传的文件
        collection_name: 集合名称
        
    Returns:
        DocumentAddResponse: 添加结果
    """
    try:
        # 确保上传目录存在
        upload_dir = FilePath("./uploads")
        upload_dir.mkdir(exist_ok=True)
        
        # 保存上传的文件
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # 添加文档到知识库
        result = kb_manager.add_documents(
            source_path=str(file_path),
            collection_name=collection_name
        )
        
        return DocumentAddResponse(
            status=result["status"],
            message=result["message"]
        )
    except Exception as e:
        logger.error(f"上传并添加文档失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"上传并添加文档失败: {str(e)}"
        )

@router.post("/query", response_model=QueryResponse)
async def query_knowledge_base(
    request: QueryRequest,
    kb_manager: KnowledgeBaseManager = Depends(get_kb_manager_dependency)
):
    """
    查询知识库
    
    Args:
        request: 查询请求
        
    Returns:
        QueryResponse: 查询结果
    """
    try:
        results = kb_manager.query(
            query_text=request.query_text,
            collection_name=request.collection_name,
            filter_metadata=request.filter_metadata,
            n_results=request.n_results
        )
        
        # 转换结果格式
        formatted_results = []
        for doc in results:
            formatted_results.append(
                DocumentContent(
                    content=doc.page_content,
                    metadata=doc.metadata
                )
            )
        
        return QueryResponse(
            results=formatted_results,
            query=request.query_text
        )
    except Exception as e:
        logger.error(f"查询知识库失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"查询知识库失败: {str(e)}"
        )

@router.get("/collections", response_model=CollectionResponse)
async def get_collections(
    kb_manager: KnowledgeBaseManager = Depends(get_kb_manager_dependency)
):
    """
    获取所有集合
    
    Returns:
        CollectionResponse: 集合列表
    """
    try:
        collections = kb_manager.get_collections()
        return CollectionResponse(collections=collections)
    except Exception as e:
        logger.error(f"获取集合列表失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取集合列表失败: {str(e)}"
        )

@router.get("/collections/{collection_name}/stats", response_model=CollectionStatsResponse)
async def get_collection_stats(
    collection_name: str = Path(..., description="集合名称"),
    kb_manager: KnowledgeBaseManager = Depends(get_kb_manager_dependency)
):
    """
    获取集合统计信息
    
    Args:
        collection_name: 集合名称
        
    Returns:
        CollectionStatsResponse: 集合统计信息
    """
    try:
        stats = kb_manager.get_collection_stats(collection_name=collection_name)
        if stats.get("status") == "error":
            return CollectionStatsResponse(
                status="error",
                message=stats.get("message", "获取集合统计信息失败")
            )
        else:
            return CollectionStatsResponse(
                status="success",
                stats=stats
            )
    except Exception as e:
        logger.error(f"获取集合统计信息失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取集合统计信息失败: {str(e)}"
        )

@router.delete("/collections/{collection_name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_collection(
    collection_name: str = Path(..., description="集合名称"),
    kb_manager: KnowledgeBaseManager = Depends(get_kb_manager_dependency)
):
    """
    删除集合
    
    Args:
        collection_name: 集合名称
    """
    try:
        result = kb_manager.delete_collection(collection_name=collection_name)
        if result.get("status") == "error":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("message", "删除集合失败")
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除集合失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除集合失败: {str(e)}"
        )
