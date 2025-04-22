"""
文档处理服务模块

提供文档加载、处理和转换的服务
"""

import logging
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Union

# 定义基础类
class Document:
    """Base class for documents"""

    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

# 尝试导入实际的文档类
try:
    from langchain_core.documents import Document
except ImportError:
    pass  # 使用上面定义的模拟类

# 定义基础加载器类
class BaseLoader:
    """Base class for document loaders"""

    def load(self):
        """Load documents"""
        raise NotImplementedError()

# 尝试导入实际的加载器类
try:
    from langchain_core.document_loaders import BaseLoader
except ImportError:
    pass  # 使用上面定义的模拟类

# 定义基础文本分割器类
class TextSplitter:
    """Base class for text splitters"""

    def split_documents(self, documents):
        """Split documents"""
        return documents

# 尝试导入实际的文本分割器
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # 如果导入失败，创建模拟类
    class RecursiveCharacterTextSplitter(TextSplitter):
        def __init__(self, chunk_size=1000, chunk_overlap=200, **kwargs):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.kwargs = kwargs

        def split_documents(self, documents):
            return documents

from src.factories.extended_document_loader_factory import ExtendedDocumentLoaderFactory
from src.config.settings import get_settings

logger = logging.getLogger(__name__)

class DocumentProcessingService:
    """文档处理服务，负责加载和处理各种格式的文档"""

    def __init__(self, text_splitter: Optional[TextSplitter] = None):
        """
        初始化文档处理服务

        Args:
            text_splitter: 文本分割器，默认为RecursiveCharacterTextSplitter
        """
        self.text_splitter = text_splitter or RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.settings = get_settings()
        self.doc_settings = self.settings.document_processing

    def load_document(self, source: str, **kwargs) -> List[Document]:
        """
        加载文档

        Args:
            source: 文档源（文件路径、URL等）
            **kwargs: 额外参数传递给加载器

        Returns:
            List[Document]: 加载的文档列表

        Raises:
            ValueError: 当文档加载失败时
        """
        try:
            loader = self._create_loader(source, **kwargs)
            documents = loader.load()
            logger.info(f"成功从 {source} 加载了 {len(documents)} 个文档")
            return documents
        except Exception as e:
            logger.error(f"加载文档 {source} 失败: {str(e)}")
            raise ValueError(f"加载文档失败: {str(e)}")

    def load_and_split_document(self, source: str, **kwargs) -> List[Document]:
        """
        加载并分割文档

        Args:
            source: 文档源（文件路径、URL等）
            **kwargs: 额外参数传递给加载器

        Returns:
            List[Document]: 分割后的文档列表

        Raises:
            ValueError: 当文档加载或分割失败时
        """
        documents = self.load_document(source, **kwargs)
        try:
            if self.doc_settings.use_parallel_processing and self._should_use_parallel_splitting(documents):
                split_documents = self.split_documents_parallel(documents)
            else:
                split_documents = self.text_splitter.split_documents(documents)
            
            logger.info(f"将 {len(documents)} 个文档分割为 {len(split_documents)} 个片段")
            return split_documents
        except Exception as e:
            logger.error(f"分割文档失败: {str(e)}")
            raise ValueError(f"分割文档失败: {str(e)}")

    def _create_loader(self, source: str, **kwargs) -> BaseLoader:
        """
        创建适当的文档加载器

        Args:
            source: 文档源（文件路径、URL等）
            **kwargs: 额外参数传递给加载器

        Returns:
            BaseLoader: 文档加载器实例

        Raises:
            ValueError: 当无法创建加载器时
        """
        # 处理URL
        if source.startswith(('http://', 'https://')):
            # 检查是否是YouTube视频
            if 'youtube.com' in source or 'youtu.be' in source:
                # 从URL中提取视频ID
                if 'youtube.com' in source and 'v=' in source:
                    video_id = source.split('v=')[1].split('&')[0]
                elif 'youtu.be' in source:
                    video_id = source.split('/')[-1].split('?')[0]
                else:
                    raise ValueError(f"无法从URL提取YouTube视频ID: {source}")

                return ExtendedDocumentLoaderFactory.create_from_youtube(video_id, **kwargs)
            else:
                return ExtendedDocumentLoaderFactory.create_from_url(source, **kwargs)

        # 处理文件路径
        elif os.path.exists(source):
            if os.path.isdir(source):
                # 处理目录
                return ExtendedDocumentLoaderFactory.create_from_directory(source, **kwargs)
            else:
                # 处理单个文件
                return ExtendedDocumentLoaderFactory.create_from_file_path(source, **kwargs)

        # 处理Git仓库
        elif source.startswith(('git://', 'git@')) or source.endswith('.git'):
            return ExtendedDocumentLoaderFactory.create_from_git(source, **kwargs)

        else:
            raise ValueError(f"无法识别的文档源: {source}")

    def get_supported_formats(self) -> List[str]:
        """
        获取支持的文档格式列表

        Returns:
            List[str]: 支持的文档格式列表
        """
        return ExtendedDocumentLoaderFactory.get_supported_file_extensions()
        
    def load_documents_parallel(self, sources: List[str], **kwargs) -> List[Document]:
        """
        并行加载多个文档

        Args:
            sources: 文档源列表（文件路径、URL等）
            **kwargs: 额外参数传递给加载器

        Returns:
            List[Document]: 加载的文档列表

        Raises:
            ValueError: 当多数文档加载失败时
        """
        if not self.doc_settings.use_parallel_processing or len(sources) < self.doc_settings.min_documents_for_parallel:
            # 对于少量文档，使用串行加载
            all_documents = []
            for source in sources:
                try:
                    docs = self.load_document(source, **kwargs)
                    all_documents.extend(docs)
                except Exception as e:
                    logger.error(f"加载文档 {source} 失败: {str(e)}")
            return all_documents
        
        # 使用线程池并行加载文档
        documents = []
        errors = []
        with ThreadPoolExecutor(max_workers=min(len(sources), self.doc_settings.max_workers)) as executor:
            future_to_source = {executor.submit(self.load_document, source, **kwargs): source for source in sources}
            for future in as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    docs = future.result()
                    documents.extend(docs)
                except Exception as e:
                    logger.error(f"并行加载文档 {source} 失败: {str(e)}")
                    errors.append((source, str(e)))
        
        # 如果所有文档都加载失败，抛出异常
        if len(documents) == 0 and len(errors) > 0:
            raise ValueError(f"所有文档加载失败: {', '.join([f'{s}: {e}' for s, e in errors])}")
        
        return documents
    
    def split_documents_parallel(self, documents: List[Document]) -> List[Document]:
        """
        并行分割文档

        Args:
            documents: 要分割的文档列表

        Returns:
            List[Document]: 分割后的文档列表
        """
        if not self.doc_settings.use_parallel_processing or not self._should_use_parallel_splitting(documents):
            # 不符合并行处理条件，使用正常的文档分割
            return self.text_splitter.split_documents(documents)
        
        # 对文档进行分组，以便并行处理
        chunk_size = max(1, len(documents) // self.doc_settings.max_workers)
        chunks = [documents[i:i+chunk_size] for i in range(0, len(documents), chunk_size)]
        
        # 使用线程池并行分割文档
        all_split_docs = []
        with ThreadPoolExecutor(max_workers=self.doc_settings.max_workers) as executor:
            future_to_chunk = {executor.submit(self.text_splitter.split_documents, chunk): i 
                               for i, chunk in enumerate(chunks)}
            for future in as_completed(future_to_chunk):
                chunk_index = future_to_chunk[future]
                try:
                    split_docs = future.result()
                    all_split_docs.extend(split_docs)
                except Exception as e:
                    logger.error(f"并行分割文档块 {chunk_index} 失败: {str(e)}")
                    # 如果并行分割失败，尝试对该块使用串行分割
                    try:
                        for doc in chunks[chunk_index]:
                            try:
                                split_docs = self.text_splitter.split_documents([doc])
                                all_split_docs.extend(split_docs)
                            except Exception as inner_e:
                                logger.error(f"分割单个文档失败: {str(inner_e)}")
                    except Exception as fallback_e:
                        logger.error(f"回退分割文档块 {chunk_index} 失败: {str(fallback_e)}")
        
        return all_split_docs
    
    async def load_document_async(self, source: str, **kwargs) -> List[Document]:
        """
        异步加载文档

        Args:
            source: 文档源（文件路径、URL等）
            **kwargs: 额外参数传递给加载器

        Returns:
            List[Document]: 加载的文档列表

        Raises:
            ValueError: 当文档加载失败时
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.load_document, source, **kwargs)
    
    async def load_and_split_document_async(self, source: str, **kwargs) -> List[Document]:
        """
        异步加载并分割文档

        Args:
            source: 文档源（文件路径、URL等）
            **kwargs: 额外参数传递给加载器

        Returns:
            List[Document]: 分割后的文档列表

        Raises:
            ValueError: 当文档加载或分割失败时
        """
        documents = await self.load_document_async(source, **kwargs)
        loop = asyncio.get_event_loop()
        
        if self.doc_settings.use_parallel_processing and self._should_use_parallel_splitting(documents):
            return await loop.run_in_executor(None, self.split_documents_parallel, documents)
        else:
            return await loop.run_in_executor(None, self.text_splitter.split_documents, documents)
    
    async def load_documents_parallel_async(self, sources: List[str], **kwargs) -> List[Document]:
        """
        异步并行加载多个文档

        Args:
            sources: 文档源列表（文件路径、URL等）
            **kwargs: 额外参数传递给加载器

        Returns:
            List[Document]: 加载的文档列表
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.load_documents_parallel, sources, **kwargs)
    
    def _should_use_parallel_splitting(self, documents: List[Document]) -> bool:
        """
        判断是否应该使用并行分割
        
        Args:
            documents: 文档列表
            
        Returns:
            bool: 是否应该使用并行分割
        """
        # 文档数量超过阈值
        if len(documents) >= self.doc_settings.min_documents_for_parallel:
            return True
        
        # 单个文档内容过大
        for doc in documents:
            if len(doc.page_content) >= self.doc_settings.chunk_size_for_parallel:
                return True
                
        return False
