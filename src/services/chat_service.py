"""
聊天服务模块

该模块提供了聊天相关的服务实现。
"""

from typing import Dict, List, Optional, Any, BinaryIO
import uuid
import datetime
import logging
from bson import ObjectId
from src.services.base import BaseService
from src.config.settings import settings
from src.services.llm_service import LLMService
from src.db.mongodb import MongoDB
from src.core.exceptions import (
    DatabaseException, LLMProviderException, ServiceInitException
)

logger = logging.getLogger(__name__)

class ChatService(BaseService):
    """聊天服务实现"""
    
    async def initialize(self):
        """初始化服务
        
        Raises:
            ServiceInitException: 服务初始化失败时抛出异常
        """
        try:
            # 初始化LLM服务
            self.llm_service = LLMService()
            await self.llm_service.initialize()
            logger.info("LLM服务初始化完成")
            
            # 确保MongoDB连接已建立
            try:
                await MongoDB.connect()
                logger.info("MongoDB连接已建立")
            except Exception as e:
                logger.error(f"MongoDB连接失败: {str(e)}")
                raise DatabaseException(f"数据库连接失败: {str(e)}")
            
            # 获取集合引用
            self.conversations_collection = await MongoDB.get_collection(
                settings.mongodb.conversations_collection
            )
            self.messages_collection = await MongoDB.get_collection(
                settings.mongodb.messages_collection
            )
        except Exception as e:
            logger.error(f"聊天服务初始化失败: {str(e)}")
            raise ServiceInitException(f"聊天服务初始化失败: {str(e)}")
    
    async def create_chat_message(self, inputs: Dict, query: str, user: str, 
                              response_mode: str = "blocking", 
                              conversation_id: Optional[str] = None,
                              files: Optional[List] = None):
        """创建聊天消息
        
        Args:
            inputs: 输入参数字典
            query: 查询文本
            user: 用户标识
            response_mode: 响应模式
            conversation_id: 会话ID
            files: 文件列表
        
        Returns:
            创建的消息结果
            
        Raises:
            DatabaseException: 数据库操作失败
            LLMProviderException: LLM调用失败
        """
        try:
            # 如果没有会话ID，创建一个新会话
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
                conversation = {
                    "_id": conversation_id,
                    "name": "新对话",
                    "user": user,
                    "created_at": datetime.datetime.now(),
                    "updated_at": datetime.datetime.now(),
                    "message_count": 0,
                    "pinned": False
                }
                await self.conversations_collection.insert_one(conversation)
                logger.info(f"为用户 {user} 创建新会话 {conversation_id}")
            else:
                # 尝试获取已存在的会话
                conversation = await self.conversations_collection.find_one({"_id": conversation_id})
                
                # 如果提供了会话ID但不存在，创建一个新会话
                if not conversation:
                    conversation = {
                        "_id": conversation_id,
                        "name": "新对话",
                        "user": user,
                        "created_at": datetime.datetime.now(),
                        "updated_at": datetime.datetime.now(),
                        "message_count": 0,
                        "pinned": False
                    }
                    await self.conversations_collection.insert_one(conversation)
                    logger.info(f"为用户 {user} 创建指定ID会话 {conversation_id}")
            
            # 创建用户消息
            user_message_id = str(uuid.uuid4())
            user_message = {
                "_id": user_message_id,
                "conversation_id": conversation_id,
                "role": "user",
                "content": query,
                "inputs": inputs,
                "created_at": datetime.datetime.now()
            }
            
            # 存储用户消息
            await self.messages_collection.insert_one(user_message)
            
            # 更新会话信息
            message_count = conversation.get("message_count", 0) + 1
            update_data = {
                "message_count": message_count,
                "updated_at": datetime.datetime.now()
            }
            
            # 如果是会话的第一条消息，以前几个词作为会话名称
            if message_count == 1:
                title_words = query.split()[:3]
                title = " ".join(title_words) + "..."
                update_data["name"] = title
            
            # 更新会话文档
            await self.conversations_collection.update_one(
                {"_id": conversation_id},
                {"$set": update_data}
            )
            
            # 获取LLM提供商和模型
            provider = inputs.get("provider", settings.DEFAULT_LLM_PROVIDER)
            model = inputs.get("model", settings.DEFAULT_LLM_MODEL)
            
            # 准备上下文
            context = []
            async for msg in self.messages_collection.find(
                {"conversation_id": conversation_id}
            ).sort("created_at", 1):
                context.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # 调用LLM生成回复
            try:
                if response_mode == "blocking":
                    # 同步模式下，等待完整回复
                    response_text = await self.llm_service.generate_text(
                        prompt=query,
                        provider=provider,
                        model=model,
                        context=context,
                        temperature=inputs.get("temperature", 0.7),
                        max_tokens=inputs.get("max_tokens", 1000)
                    )
                    
                    # 创建助手消息
                    assistant_message_id = str(uuid.uuid4())
                    assistant_message = {
                        "_id": assistant_message_id,
                        "conversation_id": conversation_id,
                        "role": "assistant",
                        "content": response_text,
                        "created_at": datetime.datetime.now()
                    }
                    
                    # 存储助手消息
                    await self.messages_collection.insert_one(assistant_message)
                    
                    # 更新会话信息
                    await self.conversations_collection.update_one(
                        {"_id": conversation_id},
                        {
                            "$inc": {"message_count": 1},
                            "$set": {"updated_at": datetime.datetime.now()}
                        }
                    )
                    
                    # 创建响应
                    response = {
                        "conversation_id": conversation_id,
                        "message_id": assistant_message_id,
                        "content": response_text,
                        "task_id": None,  # 阻塞模式下不需要任务ID
                        "created_at": assistant_message["created_at"].isoformat()
                    }
                    
                    logger.info(f"用户 {user} 在会话 {conversation_id} 中生成了同步回复")
                    return response
                else:
                    # 流式模式下，返回任务ID，实际应用中应该启动后台任务
                    task_id = str(uuid.uuid4())
                    
                    # 创建临时响应
                    response = {
                        "conversation_id": conversation_id,
                        "message_id": None,  # 流式模式下消息ID会在任务完成后创建
                        "content": None,
                        "task_id": task_id,
                        "created_at": datetime.datetime.now().isoformat()
                    }
                    
                    # 这里应该启动一个后台任务处理流式响应
                    # 在实际应用中，应使用Celery等任务队列
                    # self.start_streaming_task(task_id, conversation_id, context, query, model, inputs)
                    
                    logger.info(f"用户 {user} 在会话 {conversation_id} 中启动了流式回复任务 {task_id}")
                    return response
            except LLMProviderException as e:
                # 直接传递LLM异常
                logger.error(f"LLM调用异常: {str(e)}")
                raise e
            except Exception as e:
                # 处理错误
                error_message = f"生成回复时出错: {str(e)}"
                logger.error(error_message, exc_info=True)
                
                # 创建错误消息
                error_message_id = str(uuid.uuid4())
                error_message_doc = {
                    "_id": error_message_id,
                    "conversation_id": conversation_id,
                    "role": "system",
                    "content": error_message,
                    "created_at": datetime.datetime.now()
                }
                
                # 存储错误消息
                await self.messages_collection.insert_one(error_message_doc)
                
                # 创建错误响应
                response = {
                    "conversation_id": conversation_id,
                    "message_id": error_message_id,
                    "content": error_message,
                    "error": str(e),
                    "created_at": error_message_doc["created_at"].isoformat()
                }
                
                return response
        except DatabaseException as e:
            # 数据库异常直接传递
            logger.error(f"数据库操作异常: {str(e)}")
            raise e
        except Exception as e:
            # 未知异常转换为数据库异常
            logger.error(f"创建聊天消息时发生未知异常: {str(e)}", exc_info=True)
            raise DatabaseException(f"创建聊天消息失败: {str(e)}")
    
    async def create_streaming_chat_message(self, inputs: Dict, query: str, user: str,
                                       conversation_id: Optional[str] = None,
                                       files: Optional[List] = None):
        """创建流式聊天消息（异步生成器）
        
        Args:
            inputs: 输入参数字典
            query: 查询文本
            user: 用户标识
            conversation_id: 会话ID
            files: 文件列表
        
        Yields:
            流式响应的块
            
        Raises:
            DatabaseException: 数据库操作失败
            LLMProviderException: LLM调用失败
        """
        try:
            # 如果没有会话ID，创建一个新会话
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
                conversation = {
                    "_id": conversation_id,
                    "name": "新对话",
                    "user": user,
                    "created_at": datetime.datetime.now(),
                    "updated_at": datetime.datetime.now(),
                    "message_count": 0,
                    "pinned": False
                }
                await self.conversations_collection.insert_one(conversation)
                logger.info(f"为用户 {user} 创建新会话 {conversation_id}")
            else:
                # 尝试获取已存在的会话
                conversation = await self.conversations_collection.find_one({"_id": conversation_id})
                
                # 如果提供了会话ID但不存在，创建一个新会话
                if not conversation:
                    conversation = {
                        "_id": conversation_id,
                        "name": "新对话",
                        "user": user,
                        "created_at": datetime.datetime.now(),
                        "updated_at": datetime.datetime.now(),
                        "message_count": 0,
                        "pinned": False
                    }
                    await self.conversations_collection.insert_one(conversation)
                    logger.info(f"为用户 {user} 创建指定ID会话 {conversation_id}")
            
            # 记录用户消息
            user_message_id = str(uuid.uuid4())
            user_message = {
                "_id": user_message_id,
                "conversation_id": conversation_id,
                "role": "user",
                "content": query,
                "inputs": inputs,
                "created_at": datetime.datetime.now(),
                "user": user
            }
            await self.messages_collection.insert_one(user_message)
            
            # 更新会话信息
            await self.conversations_collection.update_one(
                {"_id": conversation_id},
                {
                    "$set": {"updated_at": datetime.datetime.now()},
                    "$inc": {"message_count": 1}
                }
            )
            
            # 初始化回答内容和ID
            answer_content = ""
            answer_id = str(uuid.uuid4())
            
            # 将文件列表传递给LLM服务（如果有）
            processed_files = []
            if files:
                # 处理文件列表
                for file_id in files:
                    # 在实际应用中，应该从存储中获取文件并处理
                    # 这里只是一个简单的替代实现
                    processed_files.append({
                        "type": "image",  # 假设文件类型，实际应该从文件元数据获取
                        "transfer_method": "local_file",  # 传输方法
                        "url": file_id  # 文件ID作为URL
                    })
            
            # 准备流式响应
            async for chunk in self.llm_service.streaming_chat_completion(
                query=query,
                user=user,
                conversation_id=conversation_id,
                inputs=inputs,
                files=processed_files
            ):
                # 更新回答内容
                if "answer" in chunk:
                    answer_content += chunk["answer"]
                
                # 产生进度块
                yield {
                    "id": answer_id,
                    "conversation_id": conversation_id,
                    "event": "message",
                    "status": "in_progress",
                    "answer": chunk.get("answer", ""),
                    "created_at": datetime.datetime.now().isoformat()
                }
            
            # 保存助手回答消息
            assistant_message = {
                "_id": answer_id,
                "conversation_id": conversation_id,
                "role": "assistant",
                "content": answer_content,
                "created_at": datetime.datetime.now(),
                "user": user
            }
            await self.messages_collection.insert_one(assistant_message)
            
            # 更新会话信息
            await self.conversations_collection.update_one(
                {"_id": conversation_id},
                {
                    "$set": {"updated_at": datetime.datetime.now()},
                    "$inc": {"message_count": 1}
                }
            )
            
            # 产生完成块
            yield {
                "id": answer_id,
                "conversation_id": conversation_id,
                "event": "message",
                "status": "complete",
                "answer": answer_content,
                "created_at": datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"创建流式聊天消息失败: {str(e)}")
            yield {
                "event": "error",
                "error": str(e)
            }
    
    async def get_conversations(self, user: str, last_id: Optional[str] = None, 
                            limit: Optional[int] = None, pinned: Optional[bool] = None):
        """获取会话列表
        
        Args:
            user: 用户标识
            last_id: 上一页最后一条记录ID
            limit: 每页记录数
            pinned: 是否只显示已固定的会话
        
        Returns:
            会话列表
        """
        # 设置默认limit
        if not limit:
            limit = 20
        
        # 构建查询条件
        query = {"user": user}
        
        # 如果需要筛选已固定的会话
        if pinned is not None:
            query["pinned"] = pinned
        
        # 如果有last_id，进行分页查询
        if last_id:
            # 获取last_id对应的记录，以获取其updated_at
            last_conversation = await self.conversations_collection.find_one(
                {"_id": last_id}
            )
            if last_conversation:
                query["updated_at"] = {"$lt": last_conversation["updated_at"]}
        
        # 执行查询
        cursor = self.conversations_collection.find(query)\
            .sort("updated_at", -1)\
            .limit(limit)
        
        # 获取结果
        conversations = []
        async for conv in cursor:
            # 转换_id为字符串
            conv["id"] = str(conv["_id"])
            del conv["_id"]
            
            # 格式化时间为ISO字符串
            if "created_at" in conv:
                conv["created_at"] = conv["created_at"].isoformat()
            if "updated_at" in conv:
                conv["updated_at"] = conv["updated_at"].isoformat()
            
            conversations.append(conv)
        
        return conversations
    
    async def get_conversation_messages(self, user: str, conversation_id: Optional[str] = None,
                                   first_id: Optional[str] = None, limit: Optional[int] = None):
        """获取会话消息
        
        Args:
            user: 用户标识
            conversation_id: 会话ID
            first_id: 第一条消息ID
            limit: 每页记录数
        
        Returns:
            会话消息列表
        """
        # 设置默认limit
        if not limit:
            limit = 20
        
        # 检查会话是否存在且属于用户
        if conversation_id:
            conversation = await self.conversations_collection.find_one({
                "_id": conversation_id,
                "user": user
            })
            if not conversation:
                return []  # 会话不存在或不属于该用户
        else:
            return []  # 未提供会话ID
        
        # 构建查询条件
        query = {"conversation_id": conversation_id}
        
        # 如果有first_id，进行分页查询
        if first_id:
            # 获取first_id对应的记录，以获取其created_at
            first_message = await self.messages_collection.find_one(
                {"_id": first_id}
            )
            if first_message:
                query["created_at"] = {"$lt": first_message["created_at"]}
        
        # 执行查询
        cursor = self.messages_collection.find(query)\
            .sort("created_at", 1)\
            .limit(limit)
        
        # 获取结果
        messages = []
        async for msg in cursor:
            # 转换_id为字符串
            msg["id"] = str(msg["_id"])
            del msg["_id"]
            
            # 格式化时间为ISO字符串
            if "created_at" in msg:
                msg["created_at"] = msg["created_at"].isoformat()
            
            messages.append(msg)
        
        return messages
    
    async def rename_conversation(self, conversation_id: str, name: str, 
                              auto_generate: bool, user: str):
        """重命名会话
        
        Args:
            conversation_id: 会话ID
            name: 新名称
            auto_generate: 是否自动生成名称
            user: 用户标识
        
        Returns:
            重命名结果
        """
        # 检查会话是否存在且属于用户
        conversation = await self.conversations_collection.find_one({
            "_id": conversation_id,
            "user": user
        })
        
        if not conversation:
            return {"success": False, "message": "会话不存在或不属于该用户"}
        
        if auto_generate:
            # 从会话的第一条消息生成名称
            first_message = await self.messages_collection.find_one(
                {"conversation_id": conversation_id, "role": "user"},
                sort=[("created_at", 1)]
            )
            
            if first_message:
                # 使用消息的前几个词作为名称
                content = first_message["content"]
                title_words = content.split()[:3]
                name = " ".join(title_words) + "..."
        
        # 更新会话名称
        result = await self.conversations_collection.update_one(
            {"_id": conversation_id, "user": user},
            {
                "$set": {
                    "name": name,
                    "updated_at": datetime.datetime.now()
                }
            }
        )
        
        if result.modified_count > 0:
            return {"success": True, "message": "会话重命名成功"}
        else:
            return {"success": False, "message": "会话重命名失败"}
    
    async def delete_conversation(self, conversation_id: str, user: str):
        """删除会话
        
        Args:
            conversation_id: 会话ID
            user: 用户标识
        
        Returns:
            删除结果
        """
        # 检查会话是否存在且属于用户
        conversation = await self.conversations_collection.find_one({
            "_id": conversation_id,
            "user": user
        })
        
        if not conversation:
            return {"success": False, "message": "会话不存在或不属于该用户"}
        
        # 删除会话
        await self.conversations_collection.delete_one({
            "_id": conversation_id, 
            "user": user
        })
        
        # 删除会话的所有消息
        await self.messages_collection.delete_many({
            "conversation_id": conversation_id
        })
        
        return {"success": True, "message": "会话删除成功"}
    
    async def stop_message(self, task_id: str, user: str):
        """停止消息生成
        
        Args:
            task_id: 任务ID
            user: 用户标识
        
        Returns:
            停止结果
        """
        # 在实际应用中，应该通过任务ID找到对应的生成任务并停止
        # 这里只是一个简单的实现
        return {"success": True, "message": "消息生成已停止"}
    
    async def message_feedback(self, message_id: str, rating: str, user: str):
        """提交消息反馈
        
        Args:
            message_id: 消息ID
            rating: 评分
            user: 用户标识
        
        Returns:
            反馈结果
        """
        # 查找消息并确认用户有权限提交反馈
        message = await self.messages_collection.find_one({"_id": message_id})
        
        if not message:
            return {"success": False, "message": "消息不存在"}
        
        # 获取消息所属的会话
        conversation = await self.conversations_collection.find_one({
            "_id": message["conversation_id"],
            "user": user
        })
        
        if not conversation:
            return {"success": False, "message": "无权对此消息提交反馈"}
        
        # 更新消息反馈
        result = await self.messages_collection.update_one(
            {"_id": message_id},
            {
                "$set": {
                    "feedback": {
                        "rating": rating,
                        "submitted_at": datetime.datetime.now(),
                        "user": user
                    }
                }
            }
        )
        
        if result.modified_count > 0:
            return {"success": True, "message": "反馈提交成功"}
        else:
            return {"success": False, "message": "反馈提交失败"}
    
    async def text_to_audio(self, text: str, user: str, streaming: bool = False):
        """文本转语音
        
        Args:
            text: 文本
            user: 用户标识
            streaming: 是否流式传输
        
        Returns:
            文本转语音结果
        """
        # 在实际应用中，应该调用TTS服务
        return {"success": True, "message": "文本转语音功能暂未实现"}
    
    async def audio_to_text(self, audio_file: BinaryIO, user: str):
        """语音转文本
        
        Args:
            audio_file: 语音文件
            user: 用户标识
        
        Returns:
            语音转文本结果
        """
        # 在实际应用中，应该调用STT服务
        return {"success": True, "message": "语音转文本功能暂未实现"}
    
    async def get_suggested(self, message_id: str, user: str):
        """获取建议的消息
        
        Args:
            message_id: 消息ID
            user: 用户标识
        
        Returns:
            建议的消息列表
        """
        # 在实际应用中，应该根据消息内容生成建议
        suggestions = [
            "请告诉我更多信息",
            "能否解释一下这个概念?",
            "有什么例子吗?"
        ]
        
        return {"success": True, "suggestions": suggestions}