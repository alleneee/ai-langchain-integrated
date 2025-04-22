"""
文本分割器工厂模块

这个模块提供了创建不同文本分割器的工厂类
"""

import logging
from typing import Dict, Any, List, Optional, Union

# 定义基础类
class TextSplitter:
    """Base class for text splitters"""

    def split_text(self, text):
        """Split text"""
        raise NotImplementedError()

# 定义语言枚举
class Language:
    """Enum for programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    GO = "go"
    RUBY = "ruby"
    CPP = "cpp"
    CSHARP = "csharp"
    PHP = "php"
    RUST = "rust"
    TYPESCRIPT = "typescript"

# 尝试导入实际的分割器
try:
    from langchain_text_splitters.base import TextSplitter
    from langchain_text_splitters import (
        RecursiveCharacterTextSplitter,
        CharacterTextSplitter,
        TokenTextSplitter,
        SentenceTransformersTokenTextSplitter,
        MarkdownTextSplitter,
        HTMLHeaderTextSplitter,
        LatexTextSplitter,
        PythonCodeTextSplitter,
        Language
    )

    # 尝试导入CodeTextSplitter
    try:
        from langchain_text_splitters import CodeTextSplitter
    except ImportError:
        # 如果不可用，使用PythonCodeTextSplitter作为替代
        class CodeTextSplitter(TextSplitter):
            def __init__(self, language=None, chunk_size=1000, chunk_overlap=0, **kwargs):
                self.language = language
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
                self.kwargs = kwargs

            def split_text(self, text):
                # 使用PythonCodeTextSplitter作为替代
                splitter = PythonCodeTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
                return splitter.split_text(text)

except ImportError:
    # 如果导入失败，创建模拟类
    class RecursiveCharacterTextSplitter(TextSplitter):
        def __init__(self, chunk_size=1000, chunk_overlap=0, separators=None, **kwargs):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.separators = separators
            self.kwargs = kwargs

        def split_text(self, text):
            return [text]

    class CharacterTextSplitter(TextSplitter):
        def __init__(self, separator="\n\n", chunk_size=1000, chunk_overlap=0, **kwargs):
            self.separator = separator
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.kwargs = kwargs

        def split_text(self, text):
            return [text]

    class TokenTextSplitter(TextSplitter):
        def __init__(self, encoding_name="cl100k_base", chunk_size=1000, chunk_overlap=0, **kwargs):
            self.encoding_name = encoding_name
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.kwargs = kwargs

        def split_text(self, text):
            return [text]

    class SentenceTransformersTokenTextSplitter(TextSplitter):
        def __init__(self, model_name="all-MiniLM-L6-v2", chunk_size=1000, chunk_overlap=0, **kwargs):
            self.model_name = model_name
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.kwargs = kwargs

        def split_text(self, text):
            return [text]

    class MarkdownTextSplitter(TextSplitter):
        def __init__(self, chunk_size=1000, chunk_overlap=0, **kwargs):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.kwargs = kwargs

        def split_text(self, text):
            return [text]

    class HTMLHeaderTextSplitter(TextSplitter):
        def __init__(self, headers_to_split_on=None, **kwargs):
            self.headers_to_split_on = headers_to_split_on or []
            self.kwargs = kwargs

        def split_text(self, text):
            return [text]

    class LatexTextSplitter(TextSplitter):
        def __init__(self, chunk_size=1000, chunk_overlap=0, **kwargs):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.kwargs = kwargs

        def split_text(self, text):
            return [text]

    class PythonCodeTextSplitter(TextSplitter):
        def __init__(self, chunk_size=1000, chunk_overlap=0, **kwargs):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.kwargs = kwargs

        def split_text(self, text):
            return [text]

    class CodeTextSplitter(TextSplitter):
        def __init__(self, language=None, chunk_size=1000, chunk_overlap=0, **kwargs):
            self.language = language
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.kwargs = kwargs

        def split_text(self, text):
            return [text]

logger = logging.getLogger(__name__)


class TextSplitterFactory:
    """文本分割器工厂类，负责根据配置创建不同的文本分割器实例"""

    @staticmethod
    def create_from_config(splitter_type: str, config: Dict[str, Any] = None) -> TextSplitter:
        """
        根据分割器类型和配置创建文本分割器

        Args:
            splitter_type: 分割器类型，例如'recursive', 'character', 'token'等
            config: 分割器配置，例如chunk_size, chunk_overlap等

        Returns:
            TextSplitter: 文本分割器实例

        Raises:
            ValueError: 当分割器类型不支持时
        """
        if config is None:
            config = {}

        splitter_type = splitter_type.lower()

        # 提取通用参数
        chunk_size = config.get("chunk_size", 1000)
        chunk_overlap = config.get("chunk_overlap", 200)

        # 根据分割器类型创建对应的实例
        if splitter_type == "recursive":
            separators = config.get("separators", None)
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators
            )

        elif splitter_type == "character":
            separator = config.get("separator", "\n\n")
            return CharacterTextSplitter(
                separator=separator,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

        elif splitter_type == "token":
            encoding_name = config.get("encoding_name", "cl100k_base")
            return TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                encoding_name=encoding_name
            )

        elif splitter_type == "sentence_transformers":
            model_name = config.get("model_name", "all-MiniLM-L6-v2")
            return SentenceTransformersTokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                model_name=model_name
            )

        elif splitter_type == "markdown":
            return MarkdownTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

        elif splitter_type == "html":
            headers_to_split_on = config.get("headers_to_split_on", [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ])
            return HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

        elif splitter_type == "latex":
            return LatexTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

        elif splitter_type == "python":
            return PythonCodeTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

        elif splitter_type == "code":
            language = config.get("language", Language.PYTHON)
            return CodeTextSplitter(
                language=language,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

        else:
            raise ValueError(f"不支持的文本分割器类型: {splitter_type}")

    @staticmethod
    def get_supported_splitter_types() -> List[str]:
        """
        获取支持的文本分割器类型列表

        Returns:
            List[str]: 支持的文本分割器类型列表
        """
        return [
            "recursive",
            "character",
            "token",
            "sentence_transformers",
            "markdown",
            "html",
            "latex",
            "python",
            "code"
        ]

    @staticmethod
    def get_supported_code_languages() -> List[str]:
        """
        获取代码分割器支持的语言列表

        Returns:
            List[str]: 支持的编程语言列表
        """
        return [language.value for language in Language]