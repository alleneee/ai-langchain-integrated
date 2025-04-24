"""
LangChain服务模块

这个模块提供了LangChain相关的服务功能，包括：
1. 与不同的LLM提供商集成
2. 向量存储处理
3. 文档加载和处理
4. 检索增强生成(RAG)实现
5. 提示模板管理
"""

import logging
import os
from typing import Dict, Any, List, Optional, Tuple, Union, Callable, Type, TypeVar, Generic, cast
import tiktoken

from langchain_core.chains import ConversationChain
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, Runnable, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.services.base import BaseService
from src.models.chat import ChatMessageRequest, ChatMessageResponse
from src.utils.token_counter import estimate_token_count

# 导入工厂类
from src.factories import (
    LLMFactory,
    EmbeddingFactory,
    VectorStoreFactory,
    DocumentLoaderFactory,
    TextSplitterFactory,
    RetrieverFactory,
    OutputParserFactory,
    PromptFactory
)

logger = logging.getLogger(__name__)

T = TypeVar('T')  # 定义泛型类型变量

class LangChainService(BaseService):
    """LangChain服务类，提供与LangChain库的集成"""
    
    def __init__(self):
        """初始化LangChain服务"""
        super().__init__()
        self._encoding = tiktoken.get_encoding("cl100k_base")
        
    async def initialize(self) -> None:
        """初始化服务"""
        # 这里可以进行必要的初始化操作
        pass
        
    @staticmethod
    def create_chat_model(provider: str, config: Dict[str, Any]) -> BaseChatModel:
        """
        创建聊天模型实例
        
        Args:
            provider: 模型提供商
            config: 模型配置
            
        Returns:
            BaseChatModel: 聊天模型实例
            
        Raises:
            ValueError: 当提供商不支持或配置无效时
        """
        try:
            # 使用LLMFactory创建模型
            return LLMFactory.create_from_config(provider, config)
        except Exception as e:
            raise ValueError(f"创建聊天模型失败: {str(e)}")
    
    @staticmethod
    def get_supported_chat_providers() -> List[str]:
        """
        获取支持的LLM提供商
        
        Returns:
            List[str]: 支持的提供商列表
        """
        return LLMFactory.get_supported_providers()
    
    @staticmethod
    def create_embeddings(provider: str, config: Dict[str, Any]) -> Embeddings:
        """
        创建嵌入模型
        
        Args:
            provider: 嵌入模型提供商
            config: 嵌入模型配置
            
        Returns:
            Embeddings: 嵌入模型实例
            
        Raises:
            ValueError: 当提供商不支持时
        """
        try:
            return EmbeddingFactory.create_from_config(provider, config)
        except Exception as e:
            raise ValueError(f"创建嵌入模型失败: {str(e)}")
    
    @staticmethod
    def get_supported_embedding_providers() -> List[str]:
        """
        获取支持的嵌入模型提供商
        
        Returns:
            List[str]: 支持的嵌入模型提供商列表
        """
        return EmbeddingFactory.get_supported_providers()
    
    @staticmethod
    def create_vector_store(
        provider: str,
        config: Dict[str, Any],
        embedding_model: Embeddings,
        collection_name: str = "default_collection"
    ) -> VectorStore:
        """
        创建向量存储
        
        Args:
            provider: 向量存储提供商
            config: 向量存储配置
            embedding_model: 嵌入模型
            collection_name: 集合名称
            
        Returns:
            VectorStore: 向量存储实例
            
        Raises:
            ValueError: 当向量存储提供商不支持时
        """
        try:
            return VectorStoreFactory.create_from_config(
                provider=provider,
                config=config,
                embedding_model=embedding_model,
                collection_name=collection_name
            )
        except Exception as e:
            raise ValueError(f"创建向量存储失败: {str(e)}")
    
    @staticmethod
    def get_supported_vector_store_providers() -> List[str]:
        """
        获取支持的向量存储提供商
        
        Returns:
            List[str]: 支持的向量存储提供商列表
        """
        return VectorStoreFactory.get_supported_providers()
    
    @staticmethod
    def create_document_loader(file_path: str, **kwargs) -> Any:
        """
        根据文件类型创建文档加载器
        
        Args:
            file_path: 文件路径
            **kwargs: 额外参数
            
        Returns:
            文档加载器实例
            
        Raises:
            ValueError: 当文件类型不支持时
        """
        try:
            return DocumentLoaderFactory.create_from_file_path(file_path, **kwargs)
        except Exception as e:
            raise ValueError(f"创建文档加载器失败: {str(e)}")
    
    @staticmethod
    def create_web_loader(url: str, **kwargs) -> Any:
        """
        创建网页加载器
        
        Args:
            url: 网页URL
            **kwargs: 额外参数
            
        Returns:
            网页加载器实例
        """
        try:
            return DocumentLoaderFactory.create_from_url(url, **kwargs)
        except Exception as e:
            raise ValueError(f"创建网页加载器失败: {str(e)}")
    
    @staticmethod
    def get_supported_file_types() -> List[str]:
        """
        获取支持的文件类型
        
        Returns:
            List[str]: 支持的文件类型列表
        """
        return DocumentLoaderFactory.get_supported_file_extensions()
    
    @staticmethod
    def create_text_splitter(splitter_type: str, config: Dict[str, Any] = None) -> Any:
        """
        创建文本分割器
        
        Args:
            splitter_type: 分割器类型
            config: 分割器配置
            
        Returns:
            文本分割器实例
            
        Raises:
            ValueError: 当分割器类型不支持时
        """
        try:
            return TextSplitterFactory.create_from_config(splitter_type, config)
        except Exception as e:
            raise ValueError(f"创建文本分割器失败: {str(e)}")
    
    @staticmethod
    def get_supported_text_splitter_types() -> List[str]:
        """
        获取支持的文本分割器类型
        
        Returns:
            List[str]: 支持的文本分割器类型列表
        """
        return TextSplitterFactory.get_supported_splitter_types()
    
    @staticmethod
    def create_retriever(
        retriever_type: str,
        vector_store: VectorStore,
        config: Dict[str, Any] = None,
        llm: Optional[BaseLanguageModel] = None
    ) -> BaseRetriever:
        """
        创建检索器
        
        Args:
            retriever_type: 检索器类型
            vector_store: 向量存储
            config: 检索器配置
            llm: 可选的语言模型，用于某些高级检索器
            
        Returns:
            BaseRetriever: 检索器实例
            
        Raises:
            ValueError: 当检索器类型不支持时
        """
        try:
            return RetrieverFactory.create_from_config(
                retriever_type=retriever_type,
                vector_store=vector_store,
                config=config,
                llm=llm
            )
        except Exception as e:
            raise ValueError(f"创建检索器失败: {str(e)}")
    
    @staticmethod
    def get_supported_retriever_types() -> List[str]:
        """
        获取支持的检索器类型
        
        Returns:
            List[str]: 支持的检索器类型列表
        """
        return RetrieverFactory.get_supported_retriever_types()
    
    @staticmethod
    def create_output_parser(parser_type: str, config: Dict[str, Any] = None) -> Any:
        """
        创建输出解析器
        
        Args:
            parser_type: 解析器类型
            config: 解析器配置
            
        Returns:
            输出解析器实例
            
        Raises:
            ValueError: 当解析器类型不支持时
        """
        try:
            return OutputParserFactory.create_from_config(parser_type, config)
        except Exception as e:
            raise ValueError(f"创建输出解析器失败: {str(e)}")
    
    @staticmethod
    def get_supported_output_parser_types() -> List[str]:
        """
        获取支持的输出解析器类型
        
        Returns:
            List[str]: 支持的输出解析器类型列表
        """
        return OutputParserFactory.get_supported_parser_types()
    
    @staticmethod
    def create_prompt_template(
        prompt_type: str,
        config: Dict[str, Any],
        output_parser: Optional[Any] = None
    ) -> Any:
        """
        创建提示模板
        
        Args:
            prompt_type: 提示模板类型
            config: 提示模板配置
            output_parser: 可选的输出解析器
            
        Returns:
            提示模板实例
            
        Raises:
            ValueError: 当提示模板类型不支持时
        """
        try:
            return PromptFactory.create_from_config(prompt_type, config, output_parser)
        except Exception as e:
            raise ValueError(f"创建提示模板失败: {str(e)}")
    
    @staticmethod
    def get_supported_prompt_types() -> List[str]:
        """
        获取支持的提示模板类型
        
        Returns:
            List[str]: 支持的提示模板类型列表
        """
        return PromptFactory.get_supported_prompt_types()
    
    @staticmethod
    def create_chat_message(type: str, content: str, **kwargs) -> Dict[str, Any]:
        """
        创建聊天消息
        
        Args:
            type: 消息类型
            content: 消息内容
            **kwargs: 额外参数
            
        Returns:
            Dict[str, Any]: 聊天消息字典
        """
        return PromptFactory.create_chat_message(type, content, **kwargs)
    
    def create_retrieval_chain(
        self,
        llm: BaseLanguageModel,
        retriever: BaseRetriever,
        prompt_template: Optional[str] = None,
        include_sources: bool = False,
        **kwargs
    ) -> Runnable:
        """
        创建检索增强生成链
        
        Args:
            llm: 语言模型
            retriever: 检索器
            prompt_template: 可选的自定义提示模板
            include_sources: 是否在输出中包含来源文档信息
            **kwargs: 其他参数
            
        Returns:
            Runnable: 检索链
        """
        # 使用默认或自定义提示模板
        if prompt_template:
            prompt = ChatPromptTemplate.from_template(prompt_template)
        else:
            # 使用更新后的提示模板格式
            prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一个有帮助的助手，使用以下上下文来回答用户的问题。"
                          "如果你在上下文中找不到答案，请说你不知道，不要编造信息。\n\n"
                          "上下文：\n{context}"),
                ("human", "{question}")
            ])

        # 如果需要包含来源
        if include_sources:
            # 创建包含来源信息的格式化器
            def format_docs_with_sources(docs: List[Document]) -> str:
                formatted_docs = []
                for i, doc in enumerate(docs):
                    source = doc.metadata.get("source", f"来源 {i+1}")
                    formatted_docs.append(f"来源[{i+1}]: {source}\n内容: {doc.page_content}")
                return "\n\n".join(formatted_docs)
                
            # 构建包含来源的链
            retrieval_chain = (
                {"context": retriever | format_docs_with_sources, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            return retrieval_chain
        
        else:
            # 创建标准RAG链
            def format_docs(docs: List[Document]) -> str:
                return "\n\n".join(f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs))
                
            # 构建新版RAG链
            retrieval_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            return retrieval_chain
            
    def _convert_to_langchain_messages(self, messages: List[Dict]) -> List[BaseMessage]:
        """
        将消息字典列表转换为LangChain消息对象列表
        
        Args:
            messages: 消息字典列表，每个消息包含role和content
            
        Returns:
            List[BaseMessage]: LangChain消息对象列表
        """
        lc_messages = []
        for message in messages:
            role = message.get("role", "user").lower()
            content = message.get("content", "")
            
            # 处理可能的工具调用内容
            tool_calls = message.get("tool_calls", [])
            tool_call_id = message.get("tool_call_id")
            
            if content:
                if role == "user":
                    lc_messages.append(HumanMessage(content=content))
                elif role == "assistant" or role == "ai":
                    # 处理可能的工具调用
                    if tool_calls:
                        # 创建带有工具调用的AI消息
                        lc_messages.append(AIMessage(content=content, tool_calls=tool_calls))
                    else:
                        lc_messages.append(AIMessage(content=content))
                elif role == "system":
                    lc_messages.append(SystemMessage(content=content))
                elif role == "tool" and tool_call_id:
                    # 添加工具消息支持
                    lc_messages.append(ToolMessage(content=content, tool_call_id=tool_call_id))
                    
        return lc_messages
    
    def process_chat(
        self,
        model: BaseChatModel,
        messages: List[Dict],
        system_message: Optional[str] = None
    ) -> Tuple[str, Dict[str, int]]:
        """
        处理聊天消息并生成回复
        
        Args:
            model: LangChain聊天模型
            messages: 消息列表
            system_message: 可选的系统消息
            
        Returns:
            Tuple[str, Dict[str, int]]: 回复内容和token使用信息
        """
        # 转换消息格式
        lc_messages = self._convert_to_langchain_messages(messages)
        
        # 如果提供了系统消息，则添加到消息列表开头
        if system_message and not any(isinstance(msg, SystemMessage) for msg in lc_messages):
            lc_messages.insert(0, SystemMessage(content=system_message))
        
        # 调用模型生成回复
        response = model.invoke(lc_messages)
        
        # 提取内容
        if hasattr(response, "content"):
            content = response.content
        else:
            content = str(response)
        
        # 计算token使用情况
        input_text = system_message or ""
        for message in messages:
            input_text += message.get("content", "") + "\n"
            
        input_tokens = len(self._encoding.encode(input_text))
        output_tokens = len(self._encoding.encode(content))
        
        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        }
        
        return content, usage

    async def process_chat_with_tools(
        self,
        model: BaseChatModel,
        messages: List[Dict],
        tools: List,
        system_message: Optional[str] = None
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        """
        处理带工具的聊天消息并生成回复
        
        Args:
            model: LangChain聊天模型
            messages: 消息列表
            tools: 工具列表
            system_message: 可选的系统消息
            
        Returns:
            Tuple[Dict[str, Any], Dict[str, int]]: 回复内容（包括可能的工具调用）和token使用信息
        """
        # 转换消息格式
        lc_messages = self._convert_to_langchain_messages(messages)
        
        # 如果提供了系统消息，则添加到消息列表开头
        if system_message and not any(isinstance(msg, SystemMessage) for msg in lc_messages):
            lc_messages.insert(0, SystemMessage(content=system_message))
        
        # 绑定工具到模型
        model_with_tools = model.bind_tools(tools)
        
        # 调用模型生成回复
        response = model_with_tools.invoke(lc_messages)
        
        # 提取内容和工具调用
        content = response.content if hasattr(response, "content") else str(response)
        tool_calls = getattr(response, "tool_calls", None)
        
        # 计算token使用情况
        input_text = system_message or ""
        for message in messages:
            input_text += message.get("content", "") + "\n"
            
        input_tokens = len(self._encoding.encode(input_text))
        output_tokens = len(self._encoding.encode(content))
        
        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        }
        
        return {
            "content": content,
            "tool_calls": tool_calls
        }, usage

    async def process_documents(
        self,
        documents: List[Document],
        splitter_type: str = "recursive",
        splitter_config: Optional[Dict[str, Any]] = None,
        embedding_provider: str = "openai",
        embedding_config: Optional[Dict[str, Any]] = None,
        vector_store_provider: str = "chroma",
        vector_store_config: Optional[Dict[str, Any]] = None
    ) -> VectorStore:
        """
        处理文档并创建向量存储
        
        Args:
            documents: 文档列表
            splitter_type: 文本分割器类型
            splitter_config: 文本分割器配置
            embedding_provider: 嵌入模型提供商
            embedding_config: 嵌入模型配置
            vector_store_provider: 向量存储提供商
            vector_store_config: 向量存储配置
            
        Returns:
            VectorStore: 向量存储实例
        """
        # 创建文本分割器
        text_splitter = self.create_text_splitter(splitter_type, splitter_config)
        
        # 分割文档
        split_documents = text_splitter.split_documents(documents)
        
        # 创建嵌入模型
        if embedding_config is None:
            embedding_config = {}
        embedding_model = self.create_embeddings(embedding_provider, embedding_config)
        
        # 创建向量存储
        if vector_store_config is None:
            vector_store_config = {}
        
        return self.create_vector_store(
            provider=vector_store_provider,
            config=vector_store_config,
            embedding_model=embedding_model
        ).from_documents(split_documents, embedding_model)
    
    async def load_and_process_documents(
        self,
        file_paths: List[str],
        **kwargs
    ) -> VectorStore:
        """
        加载并处理文档
        
        Args:
            file_paths: 文件路径列表
            **kwargs: 其他参数，传递给process_documents方法
            
        Returns:
            VectorStore: 向量存储实例
        """
        all_documents = []
        
        # 加载所有文档
        for file_path in file_paths:
            try:
                loader = self.create_document_loader(file_path)
                documents = loader.load()
                all_documents.extend(documents)
                logger.info(f"成功加载文件: {file_path}, 共 {len(documents)} 个文档")
            except Exception as e:
                logger.error(f"加载文件 {file_path} 失败: {str(e)}")
                
        # 如果没有文档，抛出异常
        if not all_documents:
            raise ValueError("没有成功加载任何文档")
            
        # 处理文档
        return await self.process_documents(all_documents, **kwargs) 