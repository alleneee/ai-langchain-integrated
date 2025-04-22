"""
补全服务模块

该模块提供了文本补全相关的服务实现。
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid

from src.services.base import BaseService
from src.config.settings import settings
from src.factories.llm_factory import LLMFactory

logger = logging.getLogger(__name__)

class CompletionService(BaseService):
    """补全服务"""

    def __init__(
        self,
        model: str = None,
        max_tokens: int = None,
        temperature: float = None
    ):
        """初始化补全服务

        Args:
            model: 模型名称
            max_tokens: 最大生成令牌数
            temperature: 温度参数
        """
        self.model = model or settings.DEFAULT_CHAT_MODEL
        self.max_tokens = max_tokens or settings.DEFAULT_MAX_TOKENS
        self.temperature = temperature or settings.DEFAULT_TEMPERATURE

        # 创建LLM实例
        try:
            self.llm = LLMFactory.create_from_model_name(
                model_name=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        except Exception as e:
            logger.error(f"创建LLM实例失败: {str(e)}")
            self.llm = None

    async def initialize(self):
        """初始化服务"""
        # 这里可以进行必要的初始化操作
        pass

    async def create_completion_message(self, inputs: Dict, response_mode: str,
                                 user: str, files: Optional[List] = None):
        """创建补全消息

        Args:
            inputs: 输入参数字典
            response_mode: 响应模式
            user: 用户标识
            files: 文件列表

        Returns:
            补全消息结果
        """
        prompt = inputs.get("prompt", "")
        if not prompt:
            return {
                "id": str(uuid.uuid4()),
                "text": "提示文本不能为空",
                "created_at": datetime.now().isoformat()
            }

        try:
            # 使用LLM生成补全
            completion_text = await self.create_completion(prompt)

            # 对于流式响应，将来需要特殊处理
            if response_mode == "streaming":
                # 实现流式响应
                pass

            return {
                "id": str(uuid.uuid4()),
                "text": completion_text,
                "created_at": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"补全生成失败: {str(e)}")
            return {
                "id": str(uuid.uuid4()),
                "text": f"补全生成失败: {str(e)}",
                "created_at": datetime.now().isoformat()
            }

    async def create_completion(self, prompt: str) -> str:
        """创建文本补全

        Args:
            prompt: 提示文本

        Returns:
            str: 补全文本

        Raises:
            Exception: 当补全生成失败时
        """
        if not self.llm:
            return "LLM实例未初始化，无法生成补全"

        try:
            # 使用LLM生成补全
            try:
                from langchain_core.messages import HumanMessage
                messages = [HumanMessage(content=prompt)]
                response = self.llm.invoke(messages)

                # 提取生成的文本
                if hasattr(response, "content"):
                    return response.content
                elif isinstance(response, dict) and "content" in response:
                    return response["content"]
                else:
                    return str(response)
            except ImportError:
                # 如果无法导入langchain_core.messages，使用简单的字符串调用
                return str(self.llm.invoke(prompt))
        except Exception as e:
            logger.error(f"补全生成失败: {str(e)}")
            raise Exception(f"补全生成失败: {str(e)}")