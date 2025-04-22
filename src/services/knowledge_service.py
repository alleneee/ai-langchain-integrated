"""
知识库服务模块

该模块提供了知识库相关的服务实现。
"""

import os
import uuid
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from src.services.base import BaseService
from src.services.langchain_service import LangChainService
from src.config.settings import settings
from src.factories import (
    EmbeddingFactory, 
    VectorStoreFactory, 
    DocumentLoaderFactory,
    TextSplitterFactory,
    RetrieverFactory
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class KnowledgeService(BaseService):
    """知识库服务"""
    
    async def initialize(self):
        """初始化服务"""
        self._langchain_service = LangChainService()
        await self._langchain_service.initialize()
        self._document_stores = {}  # 存储数据集ID与向量存储的映射
        self._embedding_models = {}  # 缓存嵌入模型
        
        # 创建知识库目录
        os.makedirs(settings.KNOWLEDGE_BASE_DIR, exist_ok=True)
    
    async def _get_or_create_embedding_model(self, provider: str, config: Dict[str, Any]) -> Any:
        """
        获取或创建嵌入模型
        
        Args:
            provider: 提供商名称
            config: 配置字典
            
        Returns:
            嵌入模型实例
        """
        cache_key = f"{provider}_{config.get('model_name', 'default')}"
        
        # 检查缓存
        if cache_key in self._embedding_models:
            return self._embedding_models[cache_key]
        
        # 创建新模型
        model = self._langchain_service.create_embeddings(provider, config)
        
        # 存入缓存
        self._embedding_models[cache_key] = model
        
        return model
    
    async def create_dataset(
        self,
        name: str,
        description: Optional[str] = None,
        embedding_provider: str = "openai",
        embedding_config: Optional[Dict[str, Any]] = None,
        vector_store_provider: str = "chroma",
        vector_store_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """创建数据集
        
        Args:
            name: 数据集名称
            description: 数据集描述
            embedding_provider: 嵌入模型提供商
            embedding_config: 嵌入模型配置
            vector_store_provider: 向量存储提供商
            vector_store_config: 向量存储配置
            
        Returns:
            创建结果
        """
        # 生成数据集ID
        dataset_id = str(uuid.uuid4())
        
        # 创建数据集目录
        dataset_dir = os.path.join(settings.KNOWLEDGE_BASE_DIR, dataset_id)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # 创建嵌入模型
        if embedding_config is None:
            embedding_config = {}
        
        embedding_model = await self._get_or_create_embedding_model(
            embedding_provider, 
            embedding_config
        )
        
        # 创建向量存储配置
        if vector_store_config is None:
            vector_store_config = {}
        
        # 设置持久化目录
        if vector_store_provider in ["chroma", "faiss"]:
            vector_store_config["persist_directory"] = os.path.join(dataset_dir, "vector_store")
        
        # 创建向量存储
        vector_store = self._langchain_service.create_vector_store(
            provider=vector_store_provider,
            config=vector_store_config,
            embedding_model=embedding_model,
            collection_name=name
        )
        
        # 保存到内存中
        self._document_stores[dataset_id] = vector_store
        
        # 创建元数据文件
        metadata = {
            "id": dataset_id,
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "embedding_provider": embedding_provider,
            "embedding_config": embedding_config,
            "vector_store_provider": vector_store_provider,
            "document_count": 0
        }
        
        # TODO: 保存元数据到文件
        
        return metadata
    
    async def get_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """获取数据集
        
        Args:
            dataset_id: 数据集ID
            
        Returns:
            数据集信息
            
        Raises:
            ValueError: 当数据集不存在时
        """
        # TODO: 从文件读取元数据
        
        # 临时返回
        return {
            "id": dataset_id,
            "name": "示例数据集",
            "description": "这是一个示例数据集",
            "created_at": datetime.now().isoformat(),
            "document_count": 0
        }
    
    async def list_datasets(self, page: Optional[int] = 1, page_size: Optional[int] = 10) -> Dict[str, Any]:
        """获取数据集列表
        
        Args:
            page: 页码
            page_size: 每页记录数
            
        Returns:
            数据集列表
        """
        # TODO: 从文件读取数据集列表
        
        # 临时返回
        return {
            "total": 0,
            "datasets": []
        }
    
    async def upload_document(
        self,
        dataset_id: str,
        file_path: str,
        splitter_type: str = "recursive",
        splitter_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """上传文档到数据集
        
        Args:
            dataset_id: 数据集ID
            file_path: 文件路径
            splitter_type: 文本分割器类型
            splitter_config: 文本分割器配置
            
        Returns:
            上传结果
            
        Raises:
            ValueError: 当数据集不存在或文件类型不支持时
        """
        try:
            # 创建文档加载器
            loader = self._langchain_service.create_document_loader(file_path)
            
            # 加载文档
            documents = loader.load()
            
            # 添加文档到数据集
            return await self.add_documents(
                dataset_id=dataset_id,
                documents=documents,
                splitter_type=splitter_type,
                splitter_config=splitter_config
            )
            
        except Exception as e:
            logger.error(f"上传文档失败: {str(e)}")
            raise ValueError(f"上传文档失败: {str(e)}")
    
    async def add_documents(
        self,
        dataset_id: str,
        documents: List[Document],
        splitter_type: str = "recursive",
        splitter_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """添加文档到数据集
        
        Args:
            dataset_id: 数据集ID
            documents: 文档列表
            splitter_type: 文本分割器类型
            splitter_config: 文本分割器配置
            
        Returns:
            添加结果
            
        Raises:
            ValueError: 当数据集不存在时
        """
        # 创建文本分割器
        text_splitter = self._langchain_service.create_text_splitter(
            splitter_type, 
            splitter_config
        )
        
        # 分割文档
        split_documents = text_splitter.split_documents(documents)
        
        # 获取向量存储
        if dataset_id not in self._document_stores:
            # TODO: 从文件加载向量存储
            raise ValueError(f"数据集不存在: {dataset_id}")
            
        vector_store = self._document_stores[dataset_id]
        
        # 添加文档到向量存储
        vector_store.add_documents(split_documents)
        
        # 更新元数据
        # TODO: 更新元数据文件
        
        return {
            "success": True,
            "document_count": len(split_documents),
            "dataset_id": dataset_id
        }
    
    async def add_texts(
        self,
        dataset_id: str,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """添加文本到数据集
        
        Args:
            dataset_id: 数据集ID
            texts: 文本列表
            metadatas: 元数据列表
            
        Returns:
            添加结果
            
        Raises:
            ValueError: 当数据集不存在时
        """
        # 获取向量存储
        if dataset_id not in self._document_stores:
            # TODO: 从文件加载向量存储
            raise ValueError(f"数据集不存在: {dataset_id}")
            
        vector_store = self._document_stores[dataset_id]
        
        # 添加文本到向量存储
        vector_store.add_texts(texts, metadatas=metadatas)
        
        # 更新元数据
        # TODO: 更新元数据文件
        
        return {
            "success": True,
            "text_count": len(texts),
            "dataset_id": dataset_id
        }
    
    async def query(
        self,
        dataset_id: str,
        query: str,
        retriever_type: str = "basic",
        retriever_config: Optional[Dict[str, Any]] = None,
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """查询数据集
        
        Args:
            dataset_id: 数据集ID
            query: 查询字符串
            retriever_type: 检索器类型
            retriever_config: 检索器配置
            top_k: 返回结果数量
            
        Returns:
            查询结果
            
        Raises:
            ValueError: 当数据集不存在时
        """
        # 获取向量存储
        if dataset_id not in self._document_stores:
            # TODO: 从文件加载向量存储
            raise ValueError(f"数据集不存在: {dataset_id}")
            
        vector_store = self._document_stores[dataset_id]
        
        # 设置检索器配置
        if retriever_config is None:
            retriever_config = {}
        
        retriever_config["top_k"] = top_k
        
        # 创建检索器
        retriever = self._langchain_service.create_retriever(
            retriever_type=retriever_type,
            vector_store=vector_store,
            config=retriever_config
        )
        
        # 执行查询
        results = retriever.get_relevant_documents(query)
        
        # 格式化结果
        formatted_results = []
        for doc in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": getattr(doc, "score", None)
            })
            
        return formatted_results
    
    async def delete_document(self, dataset_id: str, document_id: str) -> Dict[str, bool]:
        """删除文档
        
        Args:
            dataset_id: 数据集ID
            document_id: 文档ID
            
        Returns:
            删除结果
            
        Raises:
            ValueError: 当数据集不存在时
        """
        # 获取向量存储
        if dataset_id not in self._document_stores:
            # TODO: 从文件加载向量存储
            raise ValueError(f"数据集不存在: {dataset_id}")
            
        vector_store = self._document_stores[dataset_id]
        
        # 删除文档
        # TODO: 实现删除文档的逻辑，可能需要根据实际向量存储类型进行适配
        
        # 更新元数据
        # TODO: 更新元数据文件
        
        return {
            "success": True
        }
    
    async def delete_dataset(self, dataset_id: str) -> Dict[str, bool]:
        """删除数据集
        
        Args:
            dataset_id: 数据集ID
            
        Returns:
            删除结果
            
        Raises:
            ValueError: 当数据集不存在时
        """
        # 检查数据集是否存在
        if dataset_id not in self._document_stores:
            # TODO: 从文件加载向量存储
            raise ValueError(f"数据集不存在: {dataset_id}")
        
        # 从内存中移除
        del self._document_stores[dataset_id]
        
        # 删除数据集目录
        dataset_dir = os.path.join(settings.KNOWLEDGE_BASE_DIR, dataset_id)
        if os.path.exists(dataset_dir):
            import shutil
            shutil.rmtree(dataset_dir)
        
        return {
            "success": True
        } 