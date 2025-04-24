"""
连接池管理模块

该模块提供全局连接池管理功能，统一管理各类连接资源。
"""

import asyncio
import logging
from typing import Dict, Any, Optional
import aiohttp
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

logger = logging.getLogger(__name__)

class ConnectionPoolManager:
    """全局连接池管理器，采用单例模式"""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = ConnectionPoolManager()
        return cls._instance
    
    def __init__(self):
        """初始化连接池管理器"""
        # MongoDB连接池
        self.mongo_client = None
        self.mongo_db = None
        
        # HTTP会话池 - 为不同服务提供独立的会话
        self.http_sessions = {}
        
        # 连接池状态
        self.initialized = False
        self._lock = asyncio.Lock()
    
    async def initialize(self, settings):
        """初始化所有连接池
        
        Args:
            settings: 应用配置
            
        Raises:
            ConnectionError: 初始化连接池失败
        """
        # 使用锁确保只初始化一次
        async with self._lock:
            if self.initialized:
                return
            
            # 初始化MongoDB连接池
            await self._init_mongodb(settings)
            
            # 初始化默认HTTP会话
            await self._init_http_sessions()
            
            # 标记为已初始化
            self.initialized = True
            logger.info("全局连接池管理器初始化完成")
    
    async def _init_mongodb(self, settings):
        """初始化MongoDB连接池
        
        Args:
            settings: 应用配置
            
        Raises:
            ConnectionError: 初始化MongoDB连接池失败
        """
        try:
            # 配置MongoDB连接池
            self.mongo_client = AsyncIOMotorClient(
                settings.mongodb.uri,
                maxPoolSize=settings.mongodb.max_pool_size or 100,
                minPoolSize=settings.mongodb.min_pool_size or 10,
                maxIdleTimeMS=60000,  # 空闲连接最长保持时间(60秒)
                connectTimeoutMS=settings.mongodb.connect_timeout_ms or 5000,
                socketTimeoutMS=settings.mongodb.socket_timeout_ms or 30000,
                serverSelectionTimeoutMS=5000,  # 服务器选择超时
                waitQueueTimeoutMS=10000,  # 连接池满时等待队列超时
                heartbeatFrequencyMS=10000,  # 心跳检测频率
            )
            
            # 验证连接
            await self.mongo_client.admin.command('ismaster')
            
            # 获取数据库实例
            self.mongo_db = self.mongo_client[settings.mongodb.database]
            
            logger.info(f"MongoDB连接池初始化成功: {settings.mongodb.uri}")
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"MongoDB连接池初始化失败: {str(e)}")
            raise ConnectionError(f"MongoDB连接池初始化失败: {str(e)}")
    
    async def _init_http_sessions(self):
        """初始化HTTP会话池"""
        # 创建默认HTTP会话
        self.http_sessions["default"] = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60),
            connector=aiohttp.TCPConnector(
                limit=100,  # 全局默认最多100个并发连接
                keepalive_timeout=120,
                enable_cleanup_closed=True
            )
        )
        
        # 为LLM服务创建专用会话
        self.http_sessions["llm"] = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=120),  # LLM调用可能需要更长时间
            connector=aiohttp.TCPConnector(
                limit=50,  # LLM服务专用连接限制
                keepalive_timeout=180,
                enable_cleanup_closed=True
            ),
            headers={"User-Agent": "Dify-Connect/0.1.0"}
        )
        
        # 为向量存储创建专用会话
        self.http_sessions["vector_store"] = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(
                limit=20,
                keepalive_timeout=60,
                enable_cleanup_closed=True
            )
        )
        
        logger.info("HTTP会话池初始化成功")
    
    async def close(self):
        """关闭所有连接池"""
        async with self._lock:
            if not self.initialized:
                return
            
            # 关闭MongoDB连接
            if self.mongo_client:
                self.mongo_client.close()
                self.mongo_client = None
                self.mongo_db = None
                logger.info("MongoDB连接已关闭")
            
            # 关闭所有HTTP会话
            close_tasks = []
            for name, session in self.http_sessions.items():
                if not session.closed:
                    close_tasks.append(session.close())
                    logger.debug(f"开始关闭HTTP会话: {name}")
            
            if close_tasks:
                await asyncio.gather(*close_tasks)
                logger.info(f"已关闭 {len(close_tasks)} 个HTTP会话")
            
            self.http_sessions.clear()
            
            # 重置状态
            self.initialized = False
            logger.info("全局连接池管理器已关闭")
    
    async def get_mongodb(self):
        """获取MongoDB数据库实例
        
        Returns:
            AsyncIOMotorDatabase: MongoDB数据库实例
            
        Raises:
            RuntimeError: 连接池未初始化
        """
        if not self.initialized or not self.mongo_db:
            raise RuntimeError("MongoDB连接池未初始化")
        return self.mongo_db
    
    async def get_collection(self, collection_name):
        """获取MongoDB集合
        
        Args:
            collection_name: 集合名称
            
        Returns:
            AsyncIOMotorCollection: MongoDB集合
            
        Raises:
            RuntimeError: 连接池未初始化
        """
        db = await self.get_mongodb()
        return db[collection_name]
    
    def get_http_session(self, name="default") -> aiohttp.ClientSession:
        """获取HTTP会话
        
        Args:
            name: 会话名称，可选值: "default", "llm", "vector_store"或自定义名称
            
        Returns:
            aiohttp.ClientSession: HTTP会话
            
        Raises:
            RuntimeError: 连接池未初始化
        """
        if not self.initialized:
            raise RuntimeError("HTTP会话池未初始化")
        
        session = self.http_sessions.get(name)
        if not session or session.closed:
            # 如果指定名称的会话不存在或已关闭，则创建一个新会话
            logger.info(f"为 {name} 创建新HTTP会话")
            session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60),
                connector=aiohttp.TCPConnector(limit=10, enable_cleanup_closed=True)
            )
            self.http_sessions[name] = session
        
        return session 