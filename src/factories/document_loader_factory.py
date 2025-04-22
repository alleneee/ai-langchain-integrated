"""
文档加载器工厂模块

这个模块提供了创建不同文档加载器的工厂类
"""

import logging
import os
from typing import Dict, Any, List, Optional, Union

# 定义基础类
class BaseLoader:
    """Base class for document loaders"""

    def load(self):
        """Load documents"""
        raise NotImplementedError()

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
        BSHTMLLoader
    )

    # 尝试导入JSON加载器
    try:
        from langchain_community.document_loaders import UnstructuredJSONLoader
    except ImportError:
        # 如果不可用，创建模拟类
        class UnstructuredJSONLoader(BaseLoader):
            def __init__(self, file_path, **kwargs):
                self.file_path = file_path
                self.kwargs = kwargs

            def load(self):
                return []

    # 尝试导入Web加载器
    try:
        from langchain_community.document_loaders.web_base import WebBaseLoader
    except ImportError:
        class WebBaseLoader(BaseLoader):
            def __init__(self, url, **kwargs):
                self.url = url
                self.kwargs = kwargs

            def load(self):
                return []

except ImportError:
    # 如果导入失败，创建模拟类
    class TextLoader(BaseLoader):
        def __init__(self, file_path, **kwargs):
            self.file_path = file_path
            self.kwargs = kwargs

        def load(self):
            return []

    class PyPDFLoader(BaseLoader):
        def __init__(self, file_path, **kwargs):
            self.file_path = file_path
            self.kwargs = kwargs

        def load(self):
            return []

    class Docx2txtLoader(BaseLoader):
        def __init__(self, file_path, **kwargs):
            self.file_path = file_path
            self.kwargs = kwargs

        def load(self):
            return []

    class CSVLoader(BaseLoader):
        def __init__(self, file_path, **kwargs):
            self.file_path = file_path
            self.kwargs = kwargs

        def load(self):
            return []

    class UnstructuredMarkdownLoader(BaseLoader):
        def __init__(self, file_path, **kwargs):
            self.file_path = file_path
            self.kwargs = kwargs

        def load(self):
            return []

    class UnstructuredHTMLLoader(BaseLoader):
        def __init__(self, file_path, **kwargs):
            self.file_path = file_path
            self.kwargs = kwargs

        def load(self):
            return []

    class UnstructuredExcelLoader(BaseLoader):
        def __init__(self, file_path, **kwargs):
            self.file_path = file_path
            self.kwargs = kwargs

        def load(self):
            return []

    class UnstructuredPowerPointLoader(BaseLoader):
        def __init__(self, file_path, **kwargs):
            self.file_path = file_path
            self.kwargs = kwargs

        def load(self):
            return []

    class UnstructuredJSONLoader(BaseLoader):
        def __init__(self, file_path, **kwargs):
            self.file_path = file_path
            self.kwargs = kwargs

        def load(self):
            return []

    class BSHTMLLoader(BaseLoader):
        def __init__(self, file_path, **kwargs):
            self.file_path = file_path
            self.kwargs = kwargs

        def load(self):
            return []

    class WebBaseLoader(BaseLoader):
        def __init__(self, url, **kwargs):
            self.url = url
            self.kwargs = kwargs

        def load(self):
            return []

logger = logging.getLogger(__name__)


class DocumentLoaderFactory:
    """文档加载器工厂类，负责根据文件类型创建不同的文档加载器实例"""

    @staticmethod
    def create_from_file_path(file_path: str, **kwargs) -> BaseLoader:
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

        file_extension = os.path.splitext(file_path)[1].lower()

        # 根据文件扩展名选择合适的加载器
        if file_extension == '.txt':
            return TextLoader(file_path, **kwargs)

        elif file_extension == '.pdf':
            return PyPDFLoader(file_path, **kwargs)

        elif file_extension in ['.docx', '.doc']:
            return Docx2txtLoader(file_path, **kwargs)

        elif file_extension == '.csv':
            return CSVLoader(file_path, **kwargs)

        elif file_extension in ['.md', '.markdown']:
            return UnstructuredMarkdownLoader(file_path, **kwargs)

        elif file_extension in ['.html', '.htm']:
            return BSHTMLLoader(file_path, **kwargs)

        elif file_extension in ['.xlsx', '.xls']:
            return UnstructuredExcelLoader(file_path, **kwargs)

        elif file_extension in ['.pptx', '.ppt']:
            return UnstructuredPowerPointLoader(file_path, **kwargs)

        elif file_extension == '.json':
            return UnstructuredJSONLoader(file_path, **kwargs)

        else:
            raise ValueError(f"不支持的文件类型: {file_extension}")

    @staticmethod
    def create_from_url(url: str, **kwargs) -> BaseLoader:
        """
        根据URL创建网页加载器

        Args:
            url: 网页URL
            **kwargs: 额外参数传递给加载器

        Returns:
            BaseLoader: 网页加载器实例
        """
        return WebBaseLoader(url, **kwargs)

    @staticmethod
    def get_supported_file_extensions() -> List[str]:
        """
        获取支持的文件扩展名列表

        Returns:
            List[str]: 支持的文件扩展名列表
        """
        return [
            '.txt',
            '.pdf',
            '.docx', '.doc',
            '.csv',
            '.md', '.markdown',
            '.html', '.htm',
            '.xlsx', '.xls',
            '.pptx', '.ppt',
            '.json'
        ]