"""
异步向量知识库API端点模块

提供基于Celery的异步文档处理功能，适用于大型文档集合
"""

from fastapi import APIRouter, Depends, HTTPException, status, Path, Query, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import os
import logging
import tempfile
from celery.result import AsyncResult
from pathlib import Path as FilePath

from src.config.settings import get_settings
from src.celery_app.tasks.kb_tasks import add_documents_to_kb, batch_add_documents_to_kb

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter()

# --- 数据模型 ---

class AsyncAddRequest(BaseModel):
    """异步添加文档请求模型"""
    source_path: str = Field(..., description="文档源路径，可以是文件或目录")
    collection_name: Optional[str] = Field(None, description="集合名称")
    chunk_size: Optional[int] = Field(1000, description="分割大小")
    chunk_overlap: Optional[int] = Field(200, description="分割重叠大小")

class BatchAddRequest(BaseModel):
    """批量异步添加文档请求模型"""
    source_paths: List[str] = Field(..., description="文档源路径列表，每个可以是文件或目录")
    collection_name: Optional[str] = Field(None, description="集合名称")
    chunk_size: Optional[int] = Field(1000, description="分割大小")
    chunk_overlap: Optional[int] = Field(200, description="分割重叠大小")

class TaskResponse(BaseModel):
    """任务响应模型"""
    task_id: str
    status: str
    message: str

class TaskStatusResponse(BaseModel):
    """任务状态响应模型"""
    task_id: str
    status: str
    progress: Optional[int] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# --- API端点 ---

@router.post("/documents", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
async def add_documents_async(
    request: AsyncAddRequest
):
    """
    异步添加文档到知识库
    
    Args:
        request: 异步添加文档请求
        
    Returns:
        TaskResponse: 任务响应，包含任务ID
    """
    try:
        # 检查源路径是否存在
        if not os.path.exists(request.source_path):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"源路径不存在: {request.source_path}"
            )
            
        # 提交异步任务
        task = add_documents_to_kb.delay(
            source_path=request.source_path,
            collection_name=request.collection_name,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )
        
        return TaskResponse(
            task_id=task.id,
            status="PENDING",
            message="文档处理任务已提交"
        )
    except Exception as e:
        logger.error(f"提交异步添加文档任务失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"提交任务失败: {str(e)}"
        )

@router.post("/batch", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
async def batch_add_documents_async(
    request: BatchAddRequest
):
    """
    批量异步添加多个文档到知识库
    
    Args:
        request: 批量异步添加文档请求
        
    Returns:
        TaskResponse: 任务响应，包含任务ID
    """
    try:
        # 检查所有源路径是否存在
        for source_path in request.source_paths:
            if not os.path.exists(source_path):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"源路径不存在: {source_path}"
                )
            
        # 提交异步任务
        task = batch_add_documents_to_kb.delay(
            source_paths=request.source_paths,
            collection_name=request.collection_name,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )
        
        return TaskResponse(
            task_id=task.id,
            status="PENDING",
            message=f"批量文档处理任务已提交，将处理 {len(request.source_paths)} 个源"
        )
    except Exception as e:
        logger.error(f"提交批量异步添加文档任务失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"提交任务失败: {str(e)}"
        )

@router.post("/upload", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document_async(
    file: UploadFile = File(...),
    collection_name: Optional[str] = Form(None),
    chunk_size: Optional[int] = Form(1000),
    chunk_overlap: Optional[int] = Form(200)
):
    """
    异步上传并处理文档
    
    Args:
        file: 上传的文件
        collection_name: 集合名称
        chunk_size: 分割大小
        chunk_overlap: 分割重叠大小
        
    Returns:
        TaskResponse: 任务响应，包含任务ID
    """
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp:
            # 写入上传的文件内容
            content = await file.read()
            temp.write(content)
            temp_path = temp.name
        
        # 保存上传的文件到uploads目录（可选）
        upload_dir = FilePath("./uploads")
        upload_dir.mkdir(exist_ok=True)
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as f:
            f.write(content)
        
        # 提交异步任务
        task = add_documents_to_kb.delay(
            source_path=str(file_path),
            collection_name=collection_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        return TaskResponse(
            task_id=task.id,
            status="PENDING",
            message=f"文件 {file.filename} 已上传并提交处理"
        )
    except Exception as e:
        logger.error(f"异步上传并处理文档失败: {str(e)}")
        # 清理临时文件
        if 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except:
                pass
                
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"上传并处理文档失败: {str(e)}"
        )

@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str = Path(..., description="任务ID")
):
    """
    获取任务状态
    
    Args:
        task_id: 任务ID
        
    Returns:
        TaskStatusResponse: 任务状态响应
    """
    try:
        # 获取任务结果
        task_result = AsyncResult(task_id)
        
        # 构建响应
        response = TaskStatusResponse(
            task_id=task_id,
            status=task_result.status
        )
        
        # 添加进度和元数据（如果有）
        if task_result.info:
            if isinstance(task_result.info, dict):
                # 成功状态或进行中状态
                if 'current' in task_result.info and 'total' in task_result.info:
                    response.progress = int((task_result.info['current'] / task_result.info['total']) * 100)
                
                # 添加错误信息（如果有）
                if 'exc_message' in task_result.info:
                    response.error = task_result.info.get('exc_message')
                
                # 添加完整结果（如果任务成功）
                if task_result.successful():
                    response.result = task_result.info
            else:
                # 处理非字典类型的结果
                response.result = {"data": str(task_result.info)}
        
        return response
    except Exception as e:
        logger.error(f"获取任务状态失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取任务状态失败: {str(e)}"
        )
