"""
知识库处理 Celery 任务模块

定义知识库文档处理相关的异步任务
"""

import os
import logging
import time
from typing import Dict, Any, List, Optional, Union
from celery import shared_task, states
from celery.exceptions import Ignore

from src.services.knowledge_base.kb_manager import get_kb_manager
from src.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

@shared_task(bind=True, name="kb_tasks.add_documents_to_kb")
def add_documents_to_kb(self, source_path: str, collection_name: Optional[str] = None, 
                       chunk_size: int = 1000, chunk_overlap: int = 200, **kwargs) -> Dict[str, Any]:
    """
    异步将文档添加到知识库的任务
    
    Args:
        self: Celery 任务实例
        source_path: 文档源路径，可以是文件或目录
        collection_name: 集合名称，默认为配置中的默认集合
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
                'status': '正在初始化知识库管理器...'
            }
        )
        
        # 获取知识库管理器
        kb_manager = get_kb_manager()
        
        # 检查源路径是否存在
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"源路径不存在: {source_path}")
        
        # 更新任务状态
        self.update_state(
            state=states.STARTED,
            meta={
                'current': 10,
                'total': 100,
                'status': '正在准备处理文档...'
            }
        )
        
        # 处理单个文件或目录
        start_time = time.time()
        
        # 更新任务状态
        if os.path.isdir(source_path):
            logger.info(f"开始处理目录: {source_path}")
            file_count = sum(1 for _ in os.listdir(source_path) if os.path.isfile(os.path.join(source_path, _)))
            self.update_state(
                state=states.STARTED,
                meta={
                    'current': 20,
                    'total': 100,
                    'status': f'准备处理目录中的 {file_count} 个文件...'
                }
            )
        else:
            logger.info(f"开始处理文件: {source_path}")
            self.update_state(
                state=states.STARTED,
                meta={
                    'current': 20,
                    'total': 100,
                    'status': '准备处理文件...'
                }
            )
        
        # 执行添加文档操作
        self.update_state(
            state=states.STARTED,
            meta={
                'current': 30,
                'total': 100,
                'status': '正在添加文档到知识库...'
            }
        )
        
        # 设置 chunk_size 和 chunk_overlap
        kb_manager.kb.chunk_size = chunk_size
        kb_manager.kb.chunk_overlap = chunk_overlap
        
        # 添加文档到知识库
        result = kb_manager.add_documents(
            source_path=source_path,
            collection_name=collection_name
        )
        
        process_time = time.time() - start_time
        logger.info(f"文档添加完成，耗时: {process_time:.2f}秒")
        
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
            'status': result.get('status', 'success'),
            'message': result.get('message', '文档已成功添加到知识库'),
            'source_path': source_path,
            'collection_name': collection_name,
            'process_time': process_time
        }
    
    except Exception as e:
        logger.error(f"向知识库添加文档失败: {str(e)}", exc_info=True)
        
        # 更新任务状态为失败
        self.update_state(
            state=states.FAILURE,
            meta={
                'exc_type': type(e).__name__,
                'exc_message': str(e),
                'status': '向知识库添加文档失败'
            }
        )
        
        # 抛出异常，使任务状态变为失败
        raise Ignore()

@shared_task(bind=True, name="kb_tasks.batch_add_documents_to_kb")
def batch_add_documents_to_kb(self, source_paths: List[str], collection_name: Optional[str] = None,
                            chunk_size: int = 1000, chunk_overlap: int = 200, **kwargs) -> Dict[str, Any]:
    """
    异步批量将多个文档添加到知识库的任务
    
    Args:
        self: Celery 任务实例
        source_paths: 文档源路径列表，每个可以是文件或目录
        collection_name: 集合名称，默认为配置中的默认集合
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
                'status': '正在初始化知识库管理器...'
            }
        )
        
        # 获取知识库管理器
        kb_manager = get_kb_manager()
        
        # 更新任务状态
        self.update_state(
            state=states.STARTED,
            meta={
                'current': 10,
                'total': 100,
                'status': f'准备处理 {len(source_paths)} 个源...'
            }
        )
        
        # 设置 chunk_size 和 chunk_overlap
        kb_manager.kb.chunk_size = chunk_size
        kb_manager.kb.chunk_overlap = chunk_overlap
        
        # 存储所有结果
        results = []
        success_count = 0
        failed_count = 0
        
        # 处理每个源路径
        start_time = time.time()
        total_paths = len(source_paths)
        
        for i, source_path in enumerate(source_paths):
            # 更新任务状态
            progress = int(10 + (i / total_paths) * 80)
            self.update_state(
                state=states.STARTED,
                meta={
                    'current': progress,
                    'total': 100,
                    'status': f'正在处理 {i+1}/{total_paths}: {os.path.basename(source_path)}...'
                }
            )
            
            try:
                # 检查源路径是否存在
                if not os.path.exists(source_path):
                    raise FileNotFoundError(f"源路径不存在: {source_path}")
                
                # 添加文档到知识库
                path_start_time = time.time()
                result = kb_manager.add_documents(
                    source_path=source_path,
                    collection_name=collection_name
                )
                
                process_time = time.time() - path_start_time
                
                # 添加到结果列表
                results.append({
                    'source_path': source_path,
                    'status': result.get('status', 'success'),
                    'message': result.get('message', '文档已成功添加到知识库'),
                    'process_time': process_time
                })
                
                if result.get('status') == 'success':
                    success_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                logger.error(f"处理 {source_path} 失败: {str(e)}")
                results.append({
                    'source_path': source_path,
                    'status': 'error',
                    'message': f"处理失败: {str(e)}",
                    'process_time': 0
                })
                failed_count += 1
        
        total_process_time = time.time() - start_time
        logger.info(f"批量处理完成，耗时: {total_process_time:.2f}秒，成功: {success_count}，失败: {failed_count}")
        
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
            'status': 'completed',
            'message': f'批量处理完成，成功: {success_count}，失败: {failed_count}',
            'collection_name': collection_name,
            'total_sources': total_paths,
            'success_count': success_count,
            'failed_count': failed_count,
            'total_process_time': total_process_time,
            'results': results
        }
    
    except Exception as e:
        logger.error(f"批量向知识库添加文档失败: {str(e)}", exc_info=True)
        
        # 更新任务状态为失败
        self.update_state(
            state=states.FAILURE,
            meta={
                'exc_type': type(e).__name__,
                'exc_message': str(e),
                'status': '批量向知识库添加文档失败'
            }
        )
        
        # 抛出异常，使任务状态变为失败
        raise Ignore()
