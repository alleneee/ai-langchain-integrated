"""
文档处理 Celery 任务模块

定义文档处理相关的异步任务
"""

import os
import tempfile
import shutil
import logging
import time
from typing import Dict, Any, List, Optional, Union
from celery import shared_task, states
from celery.exceptions import Ignore

from src.services.document_processing_service import DocumentProcessingService
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

@shared_task(bind=True, name="process_document_from_file")
def process_document_from_file(self, file_path: str, original_filename: str, 
                              split: bool = False, chunk_size: int = 1000, 
                              chunk_overlap: int = 200, **kwargs) -> Dict[str, Any]:
    """
    处理文件文档的异步任务
    
    Args:
        self: Celery 任务实例
        file_path: 文件路径
        original_filename: 原始文件名
        split: 是否分割文档
        chunk_size: 分割大小
        chunk_overlap: 分割重叠大小
        **kwargs: 额外参数
        
    Returns:
        Dict[str, Any]: 处理结果
    """
    try:
        # 更新任务状态为进行中
        self.update_state(
            state=states.STARTED,
            meta={
                'current': 0,
                'total': 100,
                'status': '正在初始化文档处理服务...'
            }
        )
        
        # 创建文档处理服务
        service = DocumentProcessingService()
        
        # 更新任务状态
        self.update_state(
            state=states.STARTED,
            meta={
                'current': 10,
                'total': 100,
                'status': '正在加载文档...'
            }
        )
        
        # 处理文档
        if split:
            # 创建文本分割器
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            service.text_splitter = text_splitter
            
            # 更新任务状态
            self.update_state(
                state=states.STARTED,
                meta={
                    'current': 30,
                    'total': 100,
                    'status': '正在加载并分割文档...'
                }
            )
            
            # 检查是否是目录
            if os.path.isdir(file_path):
                # 加载并分割目录中的文档
                start_time = time.time()
                logger.info(f"开始并行处理目录: {file_path}")
                documents = service.load_and_split_document(file_path, **kwargs)
                logger.info(f"完成目录处理，耗时: {time.time() - start_time:.2f}秒")
            else:
                # 加载并分割单个文档
                documents = service.load_and_split_document(file_path, **kwargs)
        else:
            # 更新任务状态
            self.update_state(
                state=states.STARTED,
                meta={
                    'current': 30,
                    'total': 100,
                    'status': '正在加载文档...'
                }
            )
            
            # 检查是否是目录
            if os.path.isdir(file_path):
                # 使用并行处理加载目录中的文档
                start_time = time.time()
                logger.info(f"开始并行加载目录: {file_path}")
                documents = service.load_document(file_path, **kwargs)
                logger.info(f"完成目录加载，耗时: {time.time() - start_time:.2f}秒")
            else:
                # 加载单个文档
                documents = service.load_document(file_path, **kwargs)
        
        # 更新任务状态
        self.update_state(
            state=states.STARTED,
            meta={
                'current': 70,
                'total': 100,
                'status': '正在处理文档内容...'
            }
        )
        
        # 构建响应
        doc_contents = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in documents
        ]
        
        # 更新任务状态
        self.update_state(
            state=states.STARTED,
            meta={
                'current': 90,
                'total': 100,
                'status': '正在完成处理...'
            }
        )
        
        # 返回结果
        return {
            'status': 'success',
            'filename': original_filename,
            'document_count': len(documents),
            'documents': doc_contents
        }
    
    except Exception as e:
        logger.error(f"处理文档失败: {str(e)}", exc_info=True)
        
        # 更新任务状态为失败
        self.update_state(
            state=states.FAILURE,
            meta={
                'exc_type': type(e).__name__,
                'exc_message': str(e),
                'status': '处理文档失败'
            }
        )
        
        # 抛出异常，使任务状态变为失败
        raise Ignore()

@shared_task(bind=True, name="process_document_from_url")
def process_document_from_url(self, url: str, split: bool = False, 
                             chunk_size: int = 1000, chunk_overlap: int = 200, 
                             **kwargs) -> Dict[str, Any]:
    """
    处理URL文档的异步任务
    
    Args:
        self: Celery 任务实例
        url: 文档URL
        split: 是否分割文档
        chunk_size: 分割大小
        chunk_overlap: 分割重叠大小
        **kwargs: 额外参数
        
    Returns:
        Dict[str, Any]: 处理结果
    """
    try:
        # 更新任务状态为进行中
        self.update_state(
            state=states.STARTED,
            meta={
                'current': 0,
                'total': 100,
                'status': '正在初始化文档处理服务...'
            }
        )
        
        # 创建文档处理服务
        service = DocumentProcessingService()
        
        # 更新任务状态
        self.update_state(
            state=states.STARTED,
            meta={
                'current': 10,
                'total': 100,
                'status': '正在连接到URL...'
            }
        )
        
        # 处理文档
        if split:
            # 创建文本分割器
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            service.text_splitter = text_splitter
            
            # 更新任务状态
            self.update_state(
                state=states.STARTED,
                meta={
                    'current': 30,
                    'total': 100,
                    'status': '正在加载并分割文档...'
                }
            )
            
            # 加载并分割文档
            documents = service.load_and_split_document(url, **kwargs)
        else:
            # 更新任务状态
            self.update_state(
                state=states.STARTED,
                meta={
                    'current': 30,
                    'total': 100,
                    'status': '正在加载文档...'
                }
            )
            
            # 加载文档
            documents = service.load_document(url, **kwargs)
        
        # 更新任务状态
        self.update_state(
            state=states.STARTED,
            meta={
                'current': 70,
                'total': 100,
                'status': '正在处理文档内容...'
            }
        )
        
        # 构建响应
        doc_contents = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in documents
        ]
        
        # 更新任务状态
        self.update_state(
            state=states.STARTED,
            meta={
                'current': 90,
                'total': 100,
                'status': '正在完成处理...'
            }
        )
        
        # 返回结果
        return {
            'status': 'success',
            'filename': url,
            'document_count': len(documents),
            'documents': doc_contents
        }
    
    except Exception as e:
        logger.error(f"处理URL文档失败: {str(e)}", exc_info=True)
        
        # 更新任务状态为失败
        self.update_state(
            state=states.FAILURE,
            meta={
                'exc_type': type(e).__name__,
                'exc_message': str(e),
                'status': '处理URL文档失败'
            }
        )
        
        # 抛出异常，使任务状态变为失败
        raise Ignore()

@shared_task(bind=True, name="process_uploaded_document")
def process_uploaded_document(self, temp_file_path: str, original_filename: str, 
                             split: bool = False, chunk_size: int = 1000, 
                             chunk_overlap: int = 200, **kwargs) -> Dict[str, Any]:
    """
    处理上传文档的异步任务
    
    Args:
        self: Celery 任务实例
        temp_file_path: 临时文件路径
        original_filename: 原始文件名
        split: 是否分割文档
        chunk_size: 分割大小
        chunk_overlap: 分割重叠大小
        **kwargs: 额外参数
        
    Returns:
        Dict[str, Any]: 处理结果
    """
    try:
        # 调用文件处理任务
        result = process_document_from_file(
            self, 
            temp_file_path, 
            original_filename, 
            split, 
            chunk_size, 
            chunk_overlap, 
            **kwargs
        )
        
        # 处理完成后删除临时文件
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        
        return result
    
    except Exception as e:
        logger.error(f"处理上传文档失败: {str(e)}", exc_info=True)
        
        # 删除临时文件
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        
        # 更新任务状态为失败
        self.update_state(
            state=states.FAILURE,
            meta={
                'exc_type': type(e).__name__,
                'exc_message': str(e),
                'status': '处理上传文档失败'
            }
        )
        
        # 抛出异常，使任务状态变为失败
        raise Ignore()

@shared_task(bind=True, name="process_documents_batch")
def process_documents_batch(self, file_paths: List[str], original_filenames: List[str],
                         split: bool = False, chunk_size: int = 1000,
                         chunk_overlap: int = 200, **kwargs) -> Dict[str, Any]:
    """
    批量处理多个文档的异步任务
    
    Args:
        self: Celery 任务实例
        file_paths: 文件路径列表
        original_filenames: 原始文件名列表
        split: 是否分割文档
        chunk_size: 分割大小
        chunk_overlap: 分割重叠大小
        **kwargs: 额外参数
        
    Returns:
        Dict[str, Any]: 处理结果
    """
    try:
        # 更新任务状态为进行中
        self.update_state(
            state=states.STARTED,
            meta={
                'current': 0,
                'total': 100,
                'status': '正在初始化文档处理服务...'
            }
        )
        
        # 创建文档处理服务
        service = DocumentProcessingService()
        
        # 设置文本分割器
        if split:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            service.text_splitter = text_splitter
        
        # 更新任务状态
        self.update_state(
            state=states.STARTED,
            meta={
                'current': 10,
                'total': 100,
                'status': '正在并行加载文档...'
            }
        )
        
        # 并行处理文档
        start_time = time.time()
        logger.info(f"开始并行处理 {len(file_paths)} 个文档")
        
        if split:
            # 存储所有文档
            all_documents = []
            
            # 使用并行加载和分割
            for i, file_path in enumerate(file_paths):
                progress = int(10 + (i / len(file_paths)) * 60)
                self.update_state(
                    state=states.STARTED,
                    meta={
                        'current': progress,
                        'total': 100,
                        'status': f'正在处理文档 {i+1}/{len(file_paths)}...'
                    }
                )
                
                try:
                    docs = service.load_and_split_document(file_path, **kwargs)
                    all_documents.extend(docs)
                except Exception as e:
                    logger.error(f"处理文档 {file_path} 失败: {str(e)}")
        else:
            # 使用并行加载
            all_documents = service.load_documents_parallel(file_paths, **kwargs)
        
        logger.info(f"完成文档批量处理，耗时: {time.time() - start_time:.2f}秒，共处理 {len(all_documents)} 个文档片段")
        
        # 更新任务状态
        self.update_state(
            state=states.STARTED,
            meta={
                'current': 80,
                'total': 100,
                'status': '正在构建响应...'
            }
        )
        
        # 构建响应
        doc_contents = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in all_documents
        ]
        
        # 更新任务状态
        self.update_state(
            state=states.STARTED,
            meta={
                'current': 90,
                'total': 100,
                'status': '正在完成处理...'
            }
        )
        
        # 返回结果
        return {
            'status': 'success',
            'file_count': len(file_paths),
            'original_filenames': original_filenames,
            'document_count': len(all_documents),
            'documents': doc_contents
        }
    
    except Exception as e:
        logger.error(f"批量处理文档失败: {str(e)}", exc_info=True)
        
        # 更新任务状态为失败
        self.update_state(
            state=states.FAILURE,
            meta={
                'exc_type': type(e).__name__,
                'exc_message': str(e),
                'status': '批量处理文档失败'
            }
        )
        
        # 抛出异常，使任务状态变为失败
        raise Ignore()
