"""
MongoDB 连接模块

该模块提供MongoDB数据库连接和管理功能
"""

import logging
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from src.config.settings import settings

logger = logging.getLogger(__name__)

class MongoDB:
    """MongoDB连接管理类"""
    
    client = None
    db = None
    
    @classmethod
    async def connect(cls):
        """连接MongoDB数据库"""
        if cls.client is not None:
            return
        
        try:
            # 使用配置创建客户端连接
            cls.client = AsyncIOMotorClient(
                settings.mongodb.uri,
                maxPoolSize=settings.mongodb.max_pool_size,
                minPoolSize=settings.mongodb.min_pool_size,
                connectTimeoutMS=settings.mongodb.connect_timeout_ms,
                socketTimeoutMS=settings.mongodb.socket_timeout_ms,
            )
            
            # 验证连接
            await cls.client.admin.command('ismaster')
            logger.info(f"成功连接到MongoDB: {settings.mongodb.uri}")
            
            # 获取数据库
            cls.db = cls.client[settings.mongodb.database]
            
            # 创建索引
            await cls._create_indexes()
            
            return cls.client
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"MongoDB连接失败: {str(e)}")
            raise
    
    @classmethod
    async def disconnect(cls):
        """断开MongoDB连接"""
        if cls.client:
            cls.client.close()
            cls.client = None
            cls.db = None
            logger.info("MongoDB连接已关闭")
    
    @classmethod
    async def get_database(cls):
        """获取数据库实例"""
        if cls.db is None:
            await cls.connect()
        return cls.db
    
    @classmethod
    async def get_collection(cls, collection_name):
        """获取集合实例"""
        db = await cls.get_database()
        return db[collection_name]
    
    @classmethod
    async def _create_indexes(cls):
        """创建必要的索引"""
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
