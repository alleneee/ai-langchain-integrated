"""
输出解析器工厂模块

这个模块提供了创建不同输出解析器的工厂类
"""

import logging
import json
from typing import Dict, Any, List, Optional, Union, Type

# 定义基础类
class BaseOutputParser:
    """Base class for output parsers"""

    def parse(self, text):
        """Parse text"""
        raise NotImplementedError()

# 尝试导入实际的解析器
try:
    from langchain_core.output_parsers import (
        BaseOutputParser,
        StrOutputParser,
        JsonOutputParser,
        PydanticOutputParser,
        CommaSeparatedListOutputParser
    )

    # 尝试导入DatetimeOutputParser
    try:
        from langchain_core.output_parsers import DatetimeOutputParser
    except ImportError:
        # 如果不可用，创建模拟类
        class DatetimeOutputParser(BaseOutputParser):
            def parse(self, text):
                # 简单实现，返回原始文本
                return text
except ImportError:
    # 如果导入失败，创建模拟类
    class StrOutputParser(BaseOutputParser):
        def parse(self, text):
            return text

    class JsonOutputParser(BaseOutputParser):
        def __init__(self, pydantic_object=None):
            self.pydantic_object = pydantic_object

        def parse(self, text):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"error": "Invalid JSON"}

    class PydanticOutputParser(BaseOutputParser):
        def __init__(self, pydantic_object=None):
            self.pydantic_object = pydantic_object

        def parse(self, text):
            try:
                data = json.loads(text)
                if self.pydantic_object:
                    return self.pydantic_object(**data)
                return data
            except Exception:
                return {"error": "Parsing error"}

    class CommaSeparatedListOutputParser(BaseOutputParser):
        def parse(self, text):
            return text.split(",")

    class DatetimeOutputParser(BaseOutputParser):
        def parse(self, text):
            return text
try:
    # 尝试使用 Pydantic v2
    from pydantic import BaseModel
except ImportError:
    # 回退到 Pydantic v1 兼容层
    from pydantic import BaseModel
# 尝试导入其他解析器
try:
    from langchain_community.output_parsers import (
        ResponseSchema,
        StructuredOutputParser,
        XMLOutputParser
    )
except ImportError:
    # 如果导入失败，创建模拟类
    class ResponseSchema:
        def __init__(self, name="", description=""):
            self.name = name
            self.description = description

    class StructuredOutputParser(BaseOutputParser):
        @classmethod
        def from_response_schemas(cls, response_schemas):
            parser = cls()
            parser.response_schemas = response_schemas
            return parser

        def parse(self, text):
            # 简单实现，返回空字典
            return {}

    class XMLOutputParser(BaseOutputParser):
        def __init__(self, tags=None):
            self.tags = tags or []

        def parse(self, text):
            return text

logger = logging.getLogger(__name__)


class OutputParserFactory:
    """输出解析器工厂类，负责根据配置创建不同的输出解析器实例"""

    @staticmethod
    def create_from_config(
        parser_type: str,
        config: Dict[str, Any] = None
    ) -> BaseOutputParser:
        """
        根据解析器类型和配置创建输出解析器

        Args:
            parser_type: 解析器类型，如'str', 'json', 'pydantic', 'structured'等
            config: 解析器配置

        Returns:
            BaseOutputParser: 输出解析器实例

        Raises:
            ValueError: 当解析器类型不支持或缺少必要参数时
        """
        if config is None:
            config = {}

        parser_type = parser_type.lower()

        # 根据解析器类型创建对应的实例
        if parser_type == "str":
            return StrOutputParser()

        elif parser_type == "json":
            pydantic_schema = config.get("pydantic_schema", None)
            if pydantic_schema:
                return JsonOutputParser(pydantic_object=pydantic_schema)
            else:
                return JsonOutputParser()

        elif parser_type == "pydantic":
            pydantic_schema = config.get("pydantic_schema")
            if not pydantic_schema:
                raise ValueError("创建Pydantic输出解析器需要提供Pydantic模型类")

            return PydanticOutputParser(pydantic_object=pydantic_schema)

        elif parser_type == "structured":
            response_schemas = config.get("response_schemas")
            if not response_schemas:
                raise ValueError("创建结构化输出解析器需要提供响应模式定义")

            # 如果提供的是字典列表，转换为ResponseSchema对象
            if isinstance(response_schemas[0], dict):
                schemas = [
                    ResponseSchema(name=schema.get("name", ""), description=schema.get("description", ""))
                    for schema in response_schemas
                ]
            else:
                schemas = response_schemas

            return StructuredOutputParser.from_response_schemas(schemas)

        elif parser_type == "comma_separated_list":
            return CommaSeparatedListOutputParser()

        elif parser_type == "datetime":
            return DatetimeOutputParser()

        elif parser_type == "xml":
            tags = config.get("tags", [])
            return XMLOutputParser(tags=tags)

        else:
            raise ValueError(f"不支持的输出解析器类型: {parser_type}")

    @staticmethod
    def get_supported_parser_types() -> List[str]:
        """
        获取支持的输出解析器类型列表

        Returns:
            List[str]: 支持的输出解析器类型列表
        """
        return [
            "str",
            "json",
            "pydantic",
            "structured",
            "comma_separated_list",
            "datetime",
            "xml"
        ]

    @staticmethod
    def create_response_schemas(schema_dicts: List[Dict[str, str]]) -> List[ResponseSchema]:
        """
        从字典列表创建ResponseSchema对象列表

        Args:
            schema_dicts: 包含name和description的字典列表

        Returns:
            List[ResponseSchema]: ResponseSchema对象列表
        """
        return [
            ResponseSchema(name=schema.get("name", ""), description=schema.get("description", ""))
            for schema in schema_dicts
        ]