"""
文档处理API端点模块

提供文档加载、处理和转换的API接口
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Query
from typing import List, Dict, Any, Optional
import tempfile
import os
import shutil

from src.services.document_processing_service import DocumentProcessingService
from src.schemas.document import DocumentResponse, SupportedFormatsResponse

router = APIRouter()

@router.get("/formats", response_model=SupportedFormatsResponse)
async def get_supported_formats():
    """获取支持的文档格式列表"""
    service = DocumentProcessingService()
    formats = service.get_supported_formats()
    return SupportedFormatsResponse(formats=formats)

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    split: bool = Form(False),
    chunk_size: Optional[int] = Form(1000),
    chunk_overlap: Optional[int] = Form(200)
):
    """
    上传并处理文档
    
    Args:
        file: 上传的文件
        split: 是否分割文档
        chunk_size: 分割大小
        chunk_overlap: 分割重叠大小
        
    Returns:
        DocumentResponse: 处理结果
    """
    # 创建临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        # 写入上传的文件内容
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name
    
    try:
        # 处理文档
        service = DocumentProcessingService()
        
        if split:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            service.text_splitter = text_splitter
            documents = service.load_and_split_document(temp_file_path)
        else:
            documents = service.load_document(temp_file_path)
        
        # 构建响应
        doc_contents = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in documents
        ]
        
        return DocumentResponse(
            filename=file.filename,
            document_count=len(documents),
            documents=doc_contents
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    finally:
        # 删除临时文件
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@router.post("/process", response_model=DocumentResponse)
async def process_document(
    source: str = Form(...),
    split: bool = Form(False),
    chunk_size: Optional[int] = Form(1000),
    chunk_overlap: Optional[int] = Form(200)
):
    """
    处理文档（从URL或路径）
    
    Args:
        source: 文档源（URL或路径）
        split: 是否分割文档
        chunk_size: 分割大小
        chunk_overlap: 分割重叠大小
        
    Returns:
        DocumentResponse: 处理结果
    """
    try:
        # 处理文档
        service = DocumentProcessingService()
        
        if split:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            service.text_splitter = text_splitter
            documents = service.load_and_split_document(source)
        else:
            documents = service.load_document(source)
        
        # 构建响应
        doc_contents = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in documents
        ]
        
        return DocumentResponse(
            filename=os.path.basename(source) if os.path.exists(source) else source,
            document_count=len(documents),
            documents=doc_contents
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
