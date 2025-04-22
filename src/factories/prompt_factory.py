"""
提示模板工厂模块

这个模块提供了创建不同提示模板的工厂类
"""

import logging
import json
from typing import Dict, Any, List, Optional, Union, Callable

from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from langchain_core.output_parsers import BaseOutputParser

logger = logging.getLogger(__name__)


class PromptFactory:
    """提示模板工厂类，负责根据配置创建不同的提示模板实例"""
    
    @staticmethod
    def create_from_config(
        prompt_type: str,
        config: Dict[str, Any],
        output_parser: Optional[BaseOutputParser] = None
    ) -> Union[PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate]:
        """
        根据提示模板类型和配置创建提示模板
        
        Args:
            prompt_type: 提示模板类型，如'basic', 'chat', 'few_shot'等
            config: 提示模板配置
            output_parser: 可选的输出解析器
            
        Returns:
            Union[PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate]: 提示模板实例
            
        Raises:
            ValueError: 当提示模板类型不支持或缺少必要参数时
        """
        prompt_type = prompt_type.lower()
        
        # 根据提示模板类型创建对应的实例
        if prompt_type == "basic":
            template = config.get("template", "")
            input_variables = config.get("input_variables", [])
            
            if not template:
                raise ValueError("创建基础提示模板需要提供模板字符串")
                
            # 如果未指定输入变量，尝试从模板中提取
            if not input_variables:
                from langchain_core.prompts.prompt import extract_variables
                input_variables = list(extract_variables(template))
                
            # 创建基础提示模板
            prompt = PromptTemplate(
                template=template,
                input_variables=input_variables,
                partial_variables=config.get("partial_variables", {}),
                template_format=config.get("template_format", "f-string")
            )
            
            # 添加输出解析器（如果提供）
            if output_parser:
                prompt = prompt.pipe(output_parser)
                
            return prompt
            
        elif prompt_type == "chat":
            messages = config.get("messages", [])
            
            if not messages:
                raise ValueError("创建聊天提示模板需要提供消息列表")
                
            # 处理消息列表
            processed_messages = []
            for message in messages:
                message_type = message.get("type", "").lower()
                content = message.get("content", "")
                
                if message_type == "system":
                    processed_messages.append(
                        SystemMessagePromptTemplate.from_template(content)
                    )
                elif message_type == "human":
                    processed_messages.append(
                        HumanMessagePromptTemplate.from_template(content)
                    )
                elif message_type == "ai":
                    processed_messages.append(
                        AIMessagePromptTemplate.from_template(content)
                    )
                elif message_type == "placeholder":
                    variable_name = message.get("variable_name", "chat_history")
                    processed_messages.append(
                        MessagesPlaceholder(variable_name=variable_name)
                    )
                else:
                    logger.warning(f"未知的消息类型: {message_type}，将被忽略")
                    
            # 创建聊天提示模板
            prompt = ChatPromptTemplate.from_messages(processed_messages)
            
            # 添加输出解析器（如果提供）
            if output_parser:
                prompt = prompt.pipe(output_parser)
                
            return prompt
            
        elif prompt_type == "few_shot":
            examples = config.get("examples", [])
            example_prompt = config.get("example_prompt")
            prefix = config.get("prefix", "")
            suffix = config.get("suffix", "")
            
            if not examples:
                raise ValueError("创建少样本提示模板需要提供示例列表")
                
            if not example_prompt:
                raise ValueError("创建少样本提示模板需要提供示例提示模板")
                
            # 如果example_prompt是字典，转换为PromptTemplate
            if isinstance(example_prompt, dict):
                example_prompt = PromptTemplate(
                    template=example_prompt.get("template", ""),
                    input_variables=example_prompt.get("input_variables", [])
                )
                
            # 创建少样本提示模板
            prompt = FewShotPromptTemplate(
                examples=examples,
                example_prompt=example_prompt,
                prefix=prefix,
                suffix=suffix,
                input_variables=config.get("input_variables", []),
                example_separator=config.get("example_separator", "\n\n")
            )
            
            # 添加输出解析器（如果提供）
            if output_parser:
                prompt = prompt.pipe(output_parser)
                
            return prompt
            
        else:
            raise ValueError(f"不支持的提示模板类型: {prompt_type}")
    
    @staticmethod
    def get_supported_prompt_types() -> List[str]:
        """
        获取支持的提示模板类型列表
        
        Returns:
            List[str]: 支持的提示模板类型列表
        """
        return [
            "basic",
            "chat",
            "few_shot"
        ]
        
    @staticmethod
    def create_chat_message(type: str, content: str, **kwargs) -> Dict[str, Any]:
        """
        创建聊天消息字典
        
        Args:
            type: 消息类型，'system', 'human', 'ai'或'placeholder'
            content: 消息内容
            **kwargs: 额外参数
            
        Returns:
            Dict[str, Any]: 聊天消息字典
        """
        message = {"type": type, "content": content}
        message.update(kwargs)
        return message 