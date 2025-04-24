"""
检索器工厂模块

这个模块提供了创建不同检索器的工厂类
"""

import logging
from typing import Dict, Any, List, Optional, Union

from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_community.retrievers import (
    ContextualCompressionRetriever,
    MultiQueryRetriever,
    TimeWeightedVectorStoreRetriever,
    ParentDocumentRetriever,
    SelfQueryRetriever
)
from langchain_community.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
    LLMChainFilter
)
from langchain_core.language_models import BaseLanguageModel

logger = logging.getLogger(__name__)


class RetrieverFactory:
    """检索器工厂类，负责根据配置创建不同的检索器实例"""
    
    @staticmethod
    def create_from_config(
        retriever_type: str,
        vector_store: VectorStore,
        config: Dict[str, Any] = None,
        llm: Optional[BaseLanguageModel] = None
    ) -> BaseRetriever:
        """
        根据检索器类型和配置创建检索器
        
        Args:
            retriever_type: 检索器类型，例如'basic', 'contextual', 'multi_query'等
            vector_store: 向量存储实例
            config: 检索器配置
            llm: 可选的语言模型，用于某些高级检索器（如MultiQueryRetriever）
            
        Returns:
            BaseRetriever: 检索器实例
            
        Raises:
            ValueError: 当检索器类型不支持或缺少必要参数时
        """
        if config is None:
            config = {}
            
        retriever_type = retriever_type.lower()
        
        # 获取基础检索器
        base_retriever = vector_store.as_retriever(
            search_type=config.get("search_type", "similarity"),
            search_kwargs={
                "k": config.get("top_k", 4),
                "score_threshold": config.get("score_threshold", None),
                "filter": config.get("filter", None)
            }
        )
        
        # 根据检索器类型创建高级检索器
        if retriever_type == "basic":
            return base_retriever
            
        elif retriever_type == "contextual":
            # 确保提供了LLM
            if llm is None:
                raise ValueError("创建上下文压缩检索器需要提供语言模型")
                
            # 创建文档压缩管道
            compressors = []
            
            # 添加嵌入过滤器（如果配置）
            if config.get("use_embeddings_filter", False):
                embeddings = vector_store.embeddings
                min_similarity = config.get("min_similarity", 0.7)
                compressors.append(EmbeddingsFilter(embeddings=embeddings, similarity_threshold=min_similarity))
            
            # 添加LLM过滤器（如果配置）
            if config.get("use_llm_filter", True):
                compressors.append(LLMChainFilter.from_llm(llm))
            
            # 创建压缩管道
            if len(compressors) > 1:
                compressor = DocumentCompressorPipeline(transformers=compressors)
            elif len(compressors) == 1:
                compressor = compressors[0]
            else:
                compressor = LLMChainFilter.from_llm(llm)
            
            # 创建上下文压缩检索器
            return ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )
            
        elif retriever_type == "multi_query":
            # 确保提供了LLM
            if llm is None:
                raise ValueError("创建多查询检索器需要提供语言模型")
                
            # 创建多查询检索器
            return MultiQueryRetriever.from_llm(
                retriever=base_retriever,
                llm=llm
            )
            
        elif retriever_type == "time_weighted":
            # 创建时间加权检索器
            now_time = config.get("now_time", None)  # 可以是datetime对象
            decay_rate = config.get("decay_rate", 0.01)
            
            return TimeWeightedVectorStoreRetriever(
                vectorstore=vector_store,
                decay_rate=decay_rate,
                k=config.get("top_k", 4),
                other_score_keys=config.get("other_score_keys", ["relevance"]),
                now_time=now_time
            )
            
        else:
            raise ValueError(f"不支持的检索器类型: {retriever_type}")
    
    @staticmethod
    def get_supported_retriever_types() -> List[str]:
        """
        获取支持的检索器类型列表
        
        Returns:
            List[str]: 支持的检索器类型列表
        """
        return [
            "basic",
            "contextual",
            "multi_query",
            "time_weighted"
        ] 