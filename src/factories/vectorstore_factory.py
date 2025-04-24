"""
向量存储工厂模块

这个模块提供了创建不同向量存储的工厂类
"""

import logging
import os
from typing import Dict, Any, List, Optional

from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.vectorstores.pgvector import PGVector

logger = logging.getLogger(__name__)


class VectorStoreFactory:
    """向量存储工厂类，负责创建不同的向量存储实例"""
    
    @staticmethod
    def create_from_config(
        provider: str, 
        config: Dict[str, Any], 
        embedding_model: Embeddings,
        collection_name: str = "default_collection"
    ) -> VectorStore:
        """
        根据提供商和配置创建向量存储
        
        Args:
            provider: 向量存储提供商名称，如'chroma', 'faiss', 'pgvector'等
            config: 向量存储配置，包括持久化路径、连接信息等
            embedding_model: 用于向量化的嵌入模型
            collection_name: 集合/表名称
            
        Returns:
            VectorStore: 向量存储实例
            
        Raises:
            ValueError: 当提供商不支持或配置无效时
        """
        provider = provider.lower()
        
        # 根据提供商创建对应的向量存储
        if provider == "chroma":
            persist_directory = config.get("persist_directory", "./chroma_db")
            
            # 确保目录存在
            os.makedirs(persist_directory, exist_ok=True)
            
            return Chroma(
                collection_name=collection_name,
                embedding_function=embedding_model,
                persist_directory=persist_directory
            )
        
        elif provider == "faiss":
            persist_directory = config.get("persist_directory", "./faiss_index")
            
            # 确保目录存在
            os.makedirs(persist_directory, exist_ok=True)
            
            # 检查是否存在FAISS索引
            index_path = os.path.join(persist_directory, f"{collection_name}.faiss")
            index_exists = os.path.exists(index_path)
            
            if index_exists:
                # 从现有索引加载
                return FAISS.load_local(
                    folder_path=persist_directory,
                    embeddings=embedding_model,
                    index_name=collection_name
                )
            else:
                # 创建新的FAISS实例
                return FAISS(
                    embedding_function=embedding_model,
                    index_name=collection_name,
                    docstore_path=os.path.join(persist_directory, f"{collection_name}.pkl")
                )
        
        elif provider == "pgvector":
            connection_string = config.get("connection_string", "")
            schema_name = config.get("schema_name", "public")
            
            if not connection_string:
                raise ValueError("创建PGVector向量存储需要提供数据库连接字符串")
            
            return PGVector(
                collection_name=collection_name,
                connection_string=connection_string,
                embedding_function=embedding_model,
                schema_name=schema_name
            )
        
        else:
            raise ValueError(f"不支持的向量存储提供商: {provider}")
    
    @staticmethod
    def get_supported_providers() -> List[str]:
        """
        获取支持的向量存储提供商列表
        
        Returns:
            List[str]: 支持的提供商列表
        """
        return [
            "chroma",
            "faiss",
            "pgvector"
        ] 