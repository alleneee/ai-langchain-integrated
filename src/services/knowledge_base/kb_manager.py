"""
知识库管理模块

这个模块提供了统一的知识库管理接口，支持多种向量存储
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.services.knowledge_base.chroma_service import ChromaKnowledgeBase
from src.factories.embedding_factory import EmbeddingFactory
from src.config.settings import get_settings

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KnowledgeBaseManager:
    """知识库管理类，提供统一的接口操作不同的向量数据库"""
    
    def __init__(self):
        """初始化知识库管理器"""
        self.settings = get_settings()
        self.vector_store_type = self.settings.VECTOR_STORE_TYPE.lower()
        self.vector_store_dir = self.settings.VECTOR_STORE_DIR
        self.embedding_model = EmbeddingFactory.create_from_config(
            provider=self.settings.EMBEDDING_MODEL_PROVIDER,
            model_name=self.settings.EMBEDDING_MODEL_NAME
        )
        
        # 确保存储目录存在
        os.makedirs(self.vector_store_dir, exist_ok=True)
        
        # 初始化向量存储
        self._init_vector_store()
    
    def _init_vector_store(self):
        """初始化向量存储"""
        logger.info(f"初始化向量存储: 类型={self.vector_store_type}, 路径={self.vector_store_dir}")
        
        if self.vector_store_type == "chroma":
            self.kb = ChromaKnowledgeBase(
                persist_directory=self.vector_store_dir,
                embedding_function=self.embedding_model
            )
        elif self.vector_store_type == "faiss":
            # 如果需要使用 FAISS，需要在此添加适配代码
            raise NotImplementedError("FAISS 向量存储尚未实现")
        else:
            raise ValueError(f"不支持的向量存储类型: {self.vector_store_type}")
    
    def add_documents(self, source_path: str, collection_name: str = None) -> Dict[str, Any]:
        """
        添加文档到知识库
        
        Args:
            source_path: 文档源路径，可以是文件或目录
            collection_name: 集合名称，默认为配置中的默认集合
            
        Returns:
            Dict[str, Any]: 添加结果，包含添加的文档数等信息
        """
        if collection_name is None:
            collection_name = self.settings.DEFAULT_COLLECTION_NAME

        try:
            self.kb.add_documents(
                source_path=source_path, 
                collection_name=collection_name
            )
            return {
                "status": "success",
                "message": f"文档已添加到集合 {collection_name}"
            }
        except Exception as e:
            logger.error(f"添加文档失败: {str(e)}")
            return {
                "status": "error",
                "message": f"添加文档失败: {str(e)}"
            }
    
    def query(
        self, 
        query_text: str, 
        collection_name: str = None, 
        filter_metadata: Dict[str, Any] = None,
        n_results: int = 5
    ) -> List[Document]:
        """
        查询知识库
        
        Args:
            query_text: 查询文本
            collection_name: 集合名称，默认为配置中的默认集合
            filter_metadata: 元数据过滤条件
            n_results: 返回结果数量
            
        Returns:
            List[Document]: 查询结果文档列表
        """
        if collection_name is None:
            collection_name = self.settings.DEFAULT_COLLECTION_NAME
            
        try:
            results = self.kb.query(
                query_text=query_text,
                collection_name=collection_name,
                filter=filter_metadata,
                n_results=n_results
            )
            return results
        except Exception as e:
            logger.error(f"查询知识库失败: {str(e)}")
            return []
    
    def delete_collection(self, collection_name: str = None) -> Dict[str, Any]:
        """
        删除集合
        
        Args:
            collection_name: 要删除的集合名称，默认为配置中的默认集合
            
        Returns:
            Dict[str, Any]: 删除结果
        """
        if collection_name is None:
            collection_name = self.settings.DEFAULT_COLLECTION_NAME
            
        try:
            if hasattr(self.kb, 'delete_collection'):
                self.kb.delete_collection(collection_name=collection_name)
                return {
                    "status": "success",
                    "message": f"集合 {collection_name} 已删除"
                }
            else:
                return {
                    "status": "error",
                    "message": "当前向量存储不支持删除集合操作"
                }
        except Exception as e:
            logger.error(f"删除集合失败: {str(e)}")
            return {
                "status": "error",
                "message": f"删除集合失败: {str(e)}"
            }
    
    def get_collections(self) -> List[str]:
        """
        获取所有集合
        
        Returns:
            List[str]: 集合名称列表
        """
        try:
            if hasattr(self.kb, 'get_collections'):
                return self.kb.get_collections()
            else:
                return []
        except Exception as e:
            logger.error(f"获取集合列表失败: {str(e)}")
            return []
    
    def get_collection_stats(self, collection_name: str = None) -> Dict[str, Any]:
        """
        获取集合统计信息
        
        Args:
            collection_name: 集合名称，默认为配置中的默认集合
            
        Returns:
            Dict[str, Any]: 集合统计信息
        """
        if collection_name is None:
            collection_name = self.settings.DEFAULT_COLLECTION_NAME
            
        try:
            if hasattr(self.kb, 'get_collection_stats'):
                return self.kb.get_collection_stats(collection_name=collection_name)
            else:
                return {
                    "status": "error",
                    "message": "当前向量存储不支持获取集合统计信息"
                }
        except Exception as e:
            logger.error(f"获取集合统计信息失败: {str(e)}")
            return {
                "status": "error",
                "message": f"获取集合统计信息失败: {str(e)}"
            }


# 单例模式，全局知识库管理器
_kb_manager = None

def get_kb_manager() -> KnowledgeBaseManager:
    """
    获取知识库管理器实例（单例模式）
    
    Returns:
        KnowledgeBaseManager: 知识库管理器实例
    """
    global _kb_manager
    if _kb_manager is None:
        _kb_manager = KnowledgeBaseManager()
    return _kb_manager
