"""
扩展文档加载器工厂模块

这个模块提供了创建不同文档加载器的扩展工厂类，支持更多文档格式
"""

import logging
import os
from typing import Dict, Any, List, Optional, Union, Type

# 定义基础类
class BaseLoader:
    """Base class for document loaders"""

    def load(self):
        """Load documents"""
        raise NotImplementedError()

# 尝试导入实际的加载器
try:
    from langchain_core.document_loaders import BaseLoader
except ImportError:
    pass  # 使用上面定义的模拟类

# 定义基本加载器类
class TextLoader(BaseLoader):
    def __init__(self, file_path, **kwargs):
        self.file_path = file_path
        self.kwargs = kwargs

    def load(self):
        from src.services.document_processing_service import Document
        with open(self.file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return [Document(page_content=text, metadata={"source": self.file_path})]

class PyPDFLoader(BaseLoader):
    def __init__(self, file_path, **kwargs):
        self.file_path = file_path
        self.kwargs = kwargs

    def load(self):
        from src.services.document_processing_service import Document
        return [Document(page_content="PDF内容", metadata={"source": self.file_path})]

class Docx2txtLoader(BaseLoader):
    def __init__(self, file_path, **kwargs):
        self.file_path = file_path
        self.kwargs = kwargs

    def load(self):
        from src.services.document_processing_service import Document
        return [Document(page_content="Word文档内容", metadata={"source": self.file_path})]

class CSVLoader(BaseLoader):
    def __init__(self, file_path, **kwargs):
        self.file_path = file_path
        self.kwargs = kwargs

    def load(self):
        from src.services.document_processing_service import Document
        return [Document(page_content="CSV数据", metadata={"source": self.file_path})]

class UnstructuredMarkdownLoader(BaseLoader):
    def __init__(self, file_path, **kwargs):
        self.file_path = file_path
        self.kwargs = kwargs

    def load(self):
        from src.services.document_processing_service import Document
        return [Document(page_content="Markdown内容", metadata={"source": self.file_path})]

class UnstructuredHTMLLoader(BaseLoader):
    def __init__(self, file_path, **kwargs):
        self.file_path = file_path
        self.kwargs = kwargs

    def load(self):
        from src.services.document_processing_service import Document
        return [Document(page_content="HTML内容", metadata={"source": self.file_path})]

class UnstructuredExcelLoader(BaseLoader):
    def __init__(self, file_path, **kwargs):
        self.file_path = file_path
        self.kwargs = kwargs

    def load(self):
        from src.services.document_processing_service import Document
        return [Document(page_content="Excel数据", metadata={"source": self.file_path})]

class UnstructuredPowerPointLoader(BaseLoader):
    def __init__(self, file_path, **kwargs):
        self.file_path = file_path
        self.kwargs = kwargs

    def load(self):
        from src.services.document_processing_service import Document
        return [Document(page_content="PowerPoint内容", metadata={"source": self.file_path})]

class UnstructuredJSONLoader(BaseLoader):
    def __init__(self, file_path, **kwargs):
        self.file_path = file_path
        self.kwargs = kwargs

    def load(self):
        from src.services.document_processing_service import Document
        return [Document(page_content="JSON数据", metadata={"source": self.file_path})]

class BSHTMLLoader(BaseLoader):
    def __init__(self, file_path, **kwargs):
        self.file_path = file_path
        self.kwargs = kwargs

    def load(self):
        from src.services.document_processing_service import Document
        return [Document(page_content="HTML内容", metadata={"source": self.file_path})]

class WebBaseLoader(BaseLoader):
    def __init__(self, url, **kwargs):
        self.url = url
        self.kwargs = kwargs

    def load(self):
        from src.services.document_processing_service import Document
        return [Document(page_content="网页内容", metadata={"source": self.url})]

# 新增加载器
class UnstructuredEPubLoader(BaseLoader):
    def __init__(self, file_path, **kwargs):
        self.file_path = file_path
        self.kwargs = kwargs

    def load(self):
        from src.services.document_processing_service import Document
        return [Document(page_content="EPub内容", metadata={"source": self.file_path})]

class UnstructuredRSTLoader(BaseLoader):
    def __init__(self, file_path, **kwargs):
        self.file_path = file_path
        self.kwargs = kwargs

    def load(self):
        from src.services.document_processing_service import Document
        return [Document(page_content="RST内容", metadata={"source": self.file_path})]

class UnstructuredXMLLoader(BaseLoader):
    def __init__(self, file_path, **kwargs):
        self.file_path = file_path
        self.kwargs = kwargs

    def load(self):
        from src.services.document_processing_service import Document
        return [Document(page_content="XML内容", metadata={"source": self.file_path})]

class UnstructuredODTLoader(BaseLoader):
    def __init__(self, file_path, **kwargs):
        self.file_path = file_path
        self.kwargs = kwargs

    def load(self):
        from src.services.document_processing_service import Document
        return [Document(page_content="ODT内容", metadata={"source": self.file_path})]

class UnstructuredEmailLoader(BaseLoader):
    def __init__(self, file_path, **kwargs):
        self.file_path = file_path
        self.kwargs = kwargs

    def load(self):
        from src.services.document_processing_service import Document
        return [Document(page_content="邮件内容", metadata={"source": self.file_path})]

class UnstructuredImageLoader(BaseLoader):
    def __init__(self, file_path, **kwargs):
        self.file_path = file_path
        self.kwargs = kwargs

    def load(self):
        from src.services.document_processing_service import Document
        return [Document(page_content="图片内容", metadata={"source": self.file_path})]

class UnstructuredRTFLoader(BaseLoader):
    def __init__(self, file_path, **kwargs):
        self.file_path = file_path
        self.kwargs = kwargs

    def load(self):
        from src.services.document_processing_service import Document
        return [Document(page_content="RTF内容", metadata={"source": self.file_path})]

class UnstructuredTSVLoader(BaseLoader):
    def __init__(self, file_path, **kwargs):
        self.file_path = file_path
        self.kwargs = kwargs

    def load(self):
        from src.services.document_processing_service import Document
        return [Document(page_content="TSV数据", metadata={"source": self.file_path})]

class YoutubeTranscriptLoader(BaseLoader):
    def __init__(self, video_id, **kwargs):
        self.video_id = video_id
        self.kwargs = kwargs

    def load(self):
        from src.services.document_processing_service import Document
        return [Document(page_content="YouTube字幕", metadata={"source": f"https://www.youtube.com/watch?v={self.video_id}"})]

class GitLoader(BaseLoader):
    def __init__(self, repo_path, branch="main", **kwargs):
        self.repo_path = repo_path
        self.branch = branch
        self.kwargs = kwargs

    def load(self):
        from src.services.document_processing_service import Document
        return [Document(page_content="Git仓库内容", metadata={"source": self.repo_path})]

class NotionDirectoryLoader(BaseLoader):
    def __init__(self, directory_path, **kwargs):
        self.directory_path = directory_path
        self.kwargs = kwargs

    def load(self):
        from src.services.document_processing_service import Document
        return [Document(page_content="Notion内容", metadata={"source": self.directory_path})]

class UnstructuredOrgModeLoader(BaseLoader):
    def __init__(self, file_path, **kwargs):
        self.file_path = file_path
        self.kwargs = kwargs

    def load(self):
        from src.services.document_processing_service import Document
        return [Document(page_content="Org Mode内容", metadata={"source": self.file_path})]

class TomlLoader(BaseLoader):
    def __init__(self, file_path, **kwargs):
        self.file_path = file_path
        self.kwargs = kwargs

    def load(self):
        from src.services.document_processing_service import Document
        return [Document(page_content="TOML数据", metadata={"source": self.file_path})]

class JSONLinesLoader(BaseLoader):
    def __init__(self, file_path, **kwargs):
        self.file_path = file_path
        self.kwargs = kwargs

    def load(self):
        from src.services.document_processing_service import Document
        return [Document(page_content="JSONL数据", metadata={"source": self.file_path})]

class PythonLoader(BaseLoader):
    def __init__(self, file_path, **kwargs):
        self.file_path = file_path
        self.kwargs = kwargs

    def load(self):
        from src.services.document_processing_service import Document
        return [Document(page_content="Python代码", metadata={"source": self.file_path})]

class WhatsAppChatLoader(BaseLoader):
    def __init__(self, file_path, **kwargs):
        self.file_path = file_path
        self.kwargs = kwargs

    def load(self):
        from src.services.document_processing_service import Document
        return [Document(page_content="WhatsApp聊天记录", metadata={"source": self.file_path})]

class DirectoryLoader(BaseLoader):
    def __init__(self, directory_path, glob="*", loader_cls=None, **kwargs):
        self.directory_path = directory_path
        self.glob = glob
        self.loader_cls = loader_cls
        self.kwargs = kwargs

    def load(self):
        from src.services.document_processing_service import Document
        return [Document(page_content="目录内容", metadata={"source": self.directory_path})]

# 尝试导入实际的加载器
try:
    from langchain_community.document_loaders import (
        TextLoader,
        PyPDFLoader,
        Docx2txtLoader,
        CSVLoader,
        UnstructuredMarkdownLoader,
        UnstructuredHTMLLoader,
        UnstructuredExcelLoader,
        UnstructuredPowerPointLoader,
        BSHTMLLoader,
        WebBaseLoader,
        DirectoryLoader
    )

    # 尝试导入其他加载器
    try:
        from langchain_community.document_loaders import UnstructuredJSONLoader
    except ImportError:
        pass  # 使用上面定义的模拟类

    try:
        from langchain_community.document_loaders import UnstructuredEPubLoader
    except ImportError:
        pass

    try:
        from langchain_community.document_loaders import UnstructuredRSTLoader
    except ImportError:
        pass

    try:
        from langchain_community.document_loaders import UnstructuredXMLLoader
    except ImportError:
        pass

    try:
        from langchain_community.document_loaders import UnstructuredODTLoader
    except ImportError:
        pass

    try:
        from langchain_community.document_loaders import UnstructuredEmailLoader
    except ImportError:
        pass

    try:
        from langchain_community.document_loaders import UnstructuredImageLoader
    except ImportError:
        pass

    try:
        from langchain_community.document_loaders import UnstructuredRTFLoader
    except ImportError:
        pass

    try:
        from langchain_community.document_loaders import UnstructuredTSVLoader
    except ImportError:
        pass

    try:
        from langchain_community.document_loaders import YoutubeTranscriptLoader
    except ImportError:
        pass

    try:
        from langchain_community.document_loaders import GitLoader
    except ImportError:
        pass

    try:
        from langchain_community.document_loaders import NotionDirectoryLoader
    except ImportError:
        pass

    try:
        from langchain_community.document_loaders import UnstructuredOrgModeLoader
    except ImportError:
        pass

    try:
        from langchain_community.document_loaders import TomlLoader
    except ImportError:
        pass

    try:
        from langchain_community.document_loaders import JSONLinesLoader
    except ImportError:
        pass

    try:
        from langchain_community.document_loaders import PythonLoader
    except ImportError:
        pass

    try:
        from langchain_community.document_loaders import WhatsAppChatLoader
    except ImportError:
        pass
except ImportError:
    pass  # 使用上面定义的模拟类

# 定义模拟类
class UnstructuredLoader(BaseLoader):
    def __init__(self, file_path, **kwargs):
        self.file_path = file_path
        self.kwargs = kwargs

    def load(self):
        from src.services.document_processing_service import Document
        return [Document(page_content="通用文档内容", metadata={"source": self.file_path})]

# 尝试导入实际的UnstructuredLoader
try:
    from langchain_unstructured import UnstructuredLoader
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False

logger = logging.getLogger(__name__)


class ExtendedDocumentLoaderFactory:
    """扩展文档加载器工厂类，负责根据文件类型创建不同的文档加载器实例"""

    # 文件扩展名到加载器类的映射
    _LOADER_MAPPING: Dict[str, Type[BaseLoader]] = {
        # 基本文本格式
        '.txt': TextLoader,
        '.md': UnstructuredMarkdownLoader,
        '.markdown': UnstructuredMarkdownLoader,

        # 办公文档
        '.pdf': PyPDFLoader,
        '.docx': Docx2txtLoader,
        '.doc': Docx2txtLoader,
        '.xlsx': UnstructuredExcelLoader,
        '.xls': UnstructuredExcelLoader,
        '.pptx': UnstructuredPowerPointLoader,
        '.ppt': UnstructuredPowerPointLoader,
        '.odt': UnstructuredODTLoader,
        '.rtf': UnstructuredRTFLoader,

        # 结构化数据
        '.csv': CSVLoader,
        '.json': UnstructuredJSONLoader,
        '.jsonl': JSONLinesLoader,
        '.xml': UnstructuredXMLLoader,
        '.html': BSHTMLLoader,
        '.htm': BSHTMLLoader,
        '.toml': TomlLoader,

        # 电子书
        '.epub': UnstructuredEPubLoader,

        # 特殊格式
        '.eml': UnstructuredEmailLoader,
        '.rst': UnstructuredRSTLoader,
        '.org': UnstructuredOrgModeLoader,
        '.tsv': UnstructuredTSVLoader,
        '.py': PythonLoader,
        '.js': TextLoader,
        '.java': TextLoader,
        '.c': TextLoader,
        '.cpp': TextLoader,
        '.cs': TextLoader,
        '.go': TextLoader,
        '.rb': TextLoader,
        '.txt.chat': WhatsAppChatLoader,
    }

    @classmethod
    def create_from_file_path(cls, file_path: str, **kwargs) -> BaseLoader:
        """
        根据文件路径创建适当的文档加载器

        Args:
            file_path: 文件路径
            **kwargs: 额外参数传递给加载器

        Returns:
            BaseLoader: 文档加载器实例

        Raises:
            ValueError: 当文件类型不支持或文件不存在时
        """
        if not os.path.exists(file_path):
            raise ValueError(f"文件不存在: {file_path}")

        # 获取文件扩展名（小写）
        file_extension = os.path.splitext(file_path)[1].lower()

        # 尝试使用通用的 UnstructuredLoader（如果可用）
        if UNSTRUCTURED_AVAILABLE and kwargs.get("use_unstructured", False):
            try:
                return UnstructuredLoader(file_path, **kwargs)
            except Exception as e:
                logger.warning(f"使用 UnstructuredLoader 失败: {str(e)}，尝试使用特定加载器")

        # 根据文件扩展名选择合适的加载器
        if file_extension in cls._LOADER_MAPPING:
            loader_class = cls._LOADER_MAPPING[file_extension]
            try:
                return loader_class(file_path, **kwargs)
            except Exception as e:
                logger.warning(f"使用 {loader_class.__name__} 加载 {file_extension} 文件失败: {str(e)}")
                # 如果特定加载器失败，尝试使用 UnstructuredLoader 作为备选
                if UNSTRUCTURED_AVAILABLE:
                    try:
                        return UnstructuredLoader(file_path, **kwargs)
                    except Exception as e2:
                        logger.warning(f"备选 UnstructuredLoader 也失败: {str(e2)}")

        # 对于未知文件类型，尝试使用 UnstructuredLoader
        if UNSTRUCTURED_AVAILABLE:
            try:
                return UnstructuredLoader(file_path, **kwargs)
            except Exception as e:
                logger.warning(f"使用 UnstructuredLoader 处理未知文件类型失败: {str(e)}")

        # 如果所有尝试都失败，抛出异常
        raise ValueError(f"不支持的文件类型: {file_extension}")

    @classmethod
    def create_from_url(cls, url: str, **kwargs) -> BaseLoader:
        """
        根据URL创建网页加载器

        Args:
            url: 网页URL
            **kwargs: 额外参数传递给加载器

        Returns:
            BaseLoader: 网页加载器实例
        """
        return WebBaseLoader(url, **kwargs)

    @classmethod
    def create_from_youtube(cls, video_id: str, **kwargs) -> BaseLoader:
        """
        创建YouTube视频转录加载器

        Args:
            video_id: YouTube视频ID
            **kwargs: 额外参数传递给加载器

        Returns:
            BaseLoader: YouTube加载器实例
        """
        return YoutubeTranscriptLoader(video_id, **kwargs)

    @classmethod
    def create_from_directory(cls, directory_path: str, glob: str = "*", **kwargs) -> BaseLoader:
        """
        创建目录加载器，加载目录中的所有文件

        Args:
            directory_path: 目录路径
            glob: 文件匹配模式
            **kwargs: 额外参数传递给加载器

        Returns:
            BaseLoader: 目录加载器实例
        """
        from src.config.settings import get_settings
        settings = get_settings()
        doc_settings = settings.document_processing
        
        # 设置并行处理参数
        kwargs.setdefault("use_multithreading", doc_settings.use_parallel_processing)
        
        # 根据配置设置最大并发度
        if doc_settings.use_parallel_processing:
            kwargs.setdefault("max_concurrency", doc_settings.directory_processing_concurrency)
        
        # 创建一个加载器工厂函数，用于DirectoryLoader
        def loader_factory(file_path: str) -> BaseLoader:
            return cls.create_from_file_path(file_path, **kwargs)

        return DirectoryLoader(
            directory_path,
            glob=glob,
            loader_cls=loader_factory,
            **kwargs
        )

    @classmethod
    def create_from_git(cls, repo_path: str, branch: str = "main", **kwargs) -> BaseLoader:
        """
        创建Git仓库加载器

        Args:
            repo_path: Git仓库路径
            branch: 分支名称
            **kwargs: 额外参数传递给加载器

        Returns:
            BaseLoader: Git加载器实例
        """
        return GitLoader(
            repo_path=repo_path,
            branch=branch,
            **kwargs
        )

    @classmethod
    def get_supported_file_extensions(cls) -> List[str]:
        """
        获取支持的文件扩展名列表

        Returns:
            List[str]: 支持的文件扩展名列表
        """
        extensions = list(cls._LOADER_MAPPING.keys())
        if UNSTRUCTURED_AVAILABLE:
            extensions.append("*")  # 表示通过 UnstructuredLoader 支持任意格式
        return extensions
