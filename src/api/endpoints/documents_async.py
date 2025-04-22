"""
异步文档处理API端点模块

提供文档加载、处理和转换的异步API接口
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Query, BackgroundTasks
from typing import List, Dict, Any, Optional
import tempfile
import os
import shutil
from celery.result import AsyncResult

from src.celery_app.celery_app import app as celery_app
from src.celery_app.tasks.document_tasks import (
    process_document_from_file,
    process_document_from_url,
    process_uploaded_document
)
from src.services.document_processing_service import DocumentProcessingService
from src.schemas.document import SupportedFormatsResponse
from src.schemas.task import TaskResponse, TaskStatusResponse

router = APIRouter()

@router.get("/formats", response_model=SupportedFormatsResponse)
async def get_supported_formats():
    """获取支持的文档格式列表"""
    service = DocumentProcessingService()
    formats = service.get_supported_formats()
    return SupportedFormatsResponse(formats=formats)

@router.post("/upload", response_model=TaskResponse)
async def upload_document_async(
    file: UploadFile = File(...),
    split: bool = Form(False),
    chunk_size: Optional[int] = Form(1000),
    chunk_overlap: Optional[int] = Form(200)
):
    """
    异步上传并处理文档
    
    Args:
        file: 上传的文件
        split: 是否分割文档
        chunk_size: 分割大小
        chunk_overlap: 分割重叠大小
        
    Returns:
        TaskResponse: 任务响应
    """
    # 创建临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        # 写入上传的文件内容
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name
    
    try:
        # 提交异步任务
        task = process_uploaded_document.delay(
            temp_file_path=temp_file_path,
            original_filename=file.filename,
            split=split,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        return TaskResponse(
            task_id=task.id,
            status="PENDING",
            message="文档处理任务已提交"
        )
    
    except Exception as e:
        # 删除临时文件
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        
        raise HTTPException(status_code=500, detail=f"提交任务失败: {str(e)}")

@router.post("/process", response_model=TaskResponse)
async def process_document_async(
    source: str = Form(...),
    split: bool = Form(False),
    chunk_size: Optional[int] = Form(1000),
    chunk_overlap: Optional[int] = Form(200)
):
    """
    异步处理文档（从URL或路径）
    
    Args:
        source: 文档源（URL或路径）
        split: 是否分割文档
        chunk_size: 分割大小
        chunk_overlap: 分割重叠大小
        
    Returns:
        TaskResponse: 任务响应
    """
    try:
        # 判断是URL还是文件路径
        if source.startswith(('http://', 'https://')):
            # 提交URL处理任务
            task = process_document_from_url.delay(
                url=source,
                split=split,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        elif os.path.exists(source):
            # 提交文件处理任务
            task = process_document_from_file.delay(
                file_path=source,
                original_filename=os.path.basename(source),
                split=split,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        else:
            raise HTTPException(status_code=400, detail=f"无效的文档源: {source}")
        
        return TaskResponse(
            task_id=task.id,
            status="PENDING",
            message="文档处理任务已提交"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"提交任务失败: {str(e)}")

@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    获取任务状态
    
    Args:
        task_id: 任务ID
        
    Returns:
        TaskStatusResponse: 任务状态响应
    """
    try:
        # 获取任务结果
        task_result = AsyncResult(task_id, app=celery_app)
        
        # 获取任务状态
        if task_result.state == 'PENDING':
            # 任务等待中
            return TaskStatusResponse(
                task_id=task_id,
                status="PENDING",
                progress=0
            )
        elif task_result.state == 'STARTED':
            # 任务进行中
            meta = task_result.info or {}
            return TaskStatusResponse(
                task_id=task_id,
                status="STARTED",
                progress=meta.get('current', 0) / meta.get('total', 100) * 100 if meta.get('total') else None
            )
        elif task_result.state == 'SUCCESS':
            # 任务成功
            return TaskStatusResponse(
                task_id=task_id,
                status="SUCCESS",
                progress=100,
                result=task_result.result
            )
        elif task_result.state == 'FAILURE':
            # 任务失败
            return TaskStatusResponse(
                task_id=task_id,
                status="FAILURE",
                error=str(task_result.info)
            )
        else:
            # 其他状态
            return TaskStatusResponse(
                task_id=task_id,
                status=task_result.state
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取任务状态失败: {str(e)}")
