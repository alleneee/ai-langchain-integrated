"""
MongoDB 连接模块

该模块提供MongoDB数据库连接和管理功能，使用全局连接池管理器
"""

import logging
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from src.config.settings import settings
from src.core.connection_pool import ConnectionPoolManager

logger = logging.getLogger(__name__)

class MongoDB:
    """MongoDB连接管理类"""
    
    @classmethod
    async def connect(cls):
        """连接MongoDB数据库
        
        Returns:
            AsyncIOMotorClient: MongoDB客户端实例
            
        Raises:
            ConnectionFailure: 连接失败时抛出
        """
        try:
            # 获取连接池管理器
            pool_manager = ConnectionPoolManager.get_instance()
            
            # 初始化连接池
            await pool_manager.initialize(settings)
            
            logger.info(f"成功连接到MongoDB: {settings.mongodb.uri}")
            
            return pool_manager.mongo_client
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"MongoDB连接失败: {str(e)}")
            raise
    
    @classmethod
    async def disconnect(cls):
        """断开MongoDB连接
        
        注意：这个方法现在会关闭整个连接池，应谨慎使用
        """
        try:
            pool_manager = ConnectionPoolManager.get_instance()
            await pool_manager.close()
            logger.info("MongoDB连接已关闭")
        except Exception as e:
            logger.error(f"关闭MongoDB连接时发生错误: {str(e)}")
    
    @classmethod
    async def get_database(cls):
        """获取数据库实例
        
        Returns:
            AsyncIOMotorDatabase: MongoDB数据库实例
        """
        pool_manager = ConnectionPoolManager.get_instance()
        if not pool_manager.initialized:
            await cls.connect()
        return await pool_manager.get_mongodb()
    
    @classmethod
    async def get_collection(cls, collection_name):
        """获取集合实例
        
        Args:
            collection_name: 集合名称
            
        Returns:
            AsyncIOMotorCollection: MongoDB集合实例
        """
        pool_manager = ConnectionPoolManager.get_instance()
        if not pool_manager.initialized:
            await cls.connect()
        return await pool_manager.get_collection(collection_name)
    
    @classmethod
    async def create_indexes(cls):
        """创建必要的索引
        
        这个方法应该在应用启动时调用，以确保所有必要的索引都存在
        """
        try:
            # 会话集合索引
            conversations_collection = await cls.get_collection(settings.mongodb.conversations_collection)
            await conversations_collection.create_index([("user", 1)])
            await conversations_collection.create_index([("updated_at", -1)])
            await conversations_collection.create_index([("user", 1), ("pinned", 1), ("updated_at", -1)])
            
            # 消息集合索引
            messages_collection = await cls.get_collection(settings.mongodb.messages_collection)
            await messages_collection.create_index([("conversation_id", 1)])
            await messages_collection.create_index([("conversation_id", 1), ("created_at", 1)])
            
            logger.info("MongoDB索引创建完成")
        except Exception as e:
            logger.error(f"创建MongoDB索引时发生错误: {str(e)}")
            raise
