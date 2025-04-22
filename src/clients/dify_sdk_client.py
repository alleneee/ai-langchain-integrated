"""
Dify SDK 客户端适配器模块

该模块基于Dify官方Python SDK，提供统一的接口与Dify平台交互
"""

import logging
import asyncio
import os
from typing import Dict, Any, List, Optional, Union, Generator, Callable, AsyncGenerator

try:
    from dify import DifyClient
    DIFY_SDK_AVAILABLE = True
except ImportError:
    DIFY_SDK_AVAILABLE = False
    # 创建一个空的类以避免导入错误
    class DifyClient:
        pass

logger = logging.getLogger(__name__)

class DifySDKAdapter:
    """
    Dify SDK 适配器类，封装官方SDK提供统一接口
    """
    
    def __init__(self, api_key: str, api_base_url: Optional[str] = None):
        """
        初始化 Dify SDK 适配器
        
        Args:
            api_key: Dify API密钥
            api_base_url: Dify API基础URL (可选，如果不提供则使用官方API地址)
        
        Raises:
            ImportError: 当Dify SDK未安装时
        """
        if not DIFY_SDK_AVAILABLE:
            logger.error("Dify SDK未安装，请使用 'pip install dify' 安装官方SDK")
            raise ImportError("Dify SDK未安装，请使用 'pip install dify' 安装官方SDK")
        
        # 初始化Dify客户端
        self.client = DifyClient(api_key=api_key, base_url=api_base_url)
        self.api_key = api_key
        self.api_base_url = api_base_url
    
    def chat_completion(self, 
                       query: str, 
                       user: str = "default", 
                       conversation_id: Optional[str] = None,
                       inputs: Optional[Dict[str, Any]] = None,
                       response_mode: str = "streaming",
                       files: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        发送聊天消息（阻塞模式）
        
        Args:
            query: 用户输入/提问内容
            user: 用户标识
            conversation_id: 会话ID（可选）
            inputs: 输入变量，默认{}
            response_mode: 响应模式 (streaming或blocking)
            files: 文件列表，格式为:
                  [{"type": "image", "transfer_method": "local_file", "upload_file_id": "file_id"}]
                  或
                  [{"type": "image", "transfer_method": "remote_url", "url": "image_url"}]
            
        Returns:
            Dict[str, Any]: 聊天响应结果
        """
        # 处理文件参数
        sdk_files = None
        if files:
            sdk_files = []
            for file in files:
                if file["transfer_method"] == "local_file":
                    sdk_files.append({
                        "type": file["type"],
                        "transfer_method": "local_file",
                        "upload_file_id": file["upload_file_id"]
                    })
                elif file["transfer_method"] == "remote_url":
                    sdk_files.append({
                        "type": file["type"],
                        "transfer_method": "remote_url",
                        "url": file["url"]
                    })
        
        # 使用SDK调用聊天接口
        response = self.client.chat_completion(
            inputs=inputs or {},
            query=query,
            user=user,
            response_mode="blocking" if response_mode == "blocking" else "streaming",
            conversation_id=conversation_id,
            files=sdk_files
        )
        
        # 如果是阻塞模式直接返回结果
        if response_mode == "blocking":
            return response
        
        # 如果是流式模式，收集完整响应
        return self._collect_streaming_response(response)
    
    def streaming_chat_completion(self, 
                                query: str, 
                                user: str = "default", 
                                conversation_id: Optional[str] = None,
                                inputs: Optional[Dict[str, Any]] = None,
                                files: Optional[List[Dict[str, Any]]] = None) -> Generator[Dict[str, Any], None, None]:
        """
        发送流式聊天消息（生成器模式）
        
        Args:
            query: 用户输入/提问内容
            user: 用户标识
            conversation_id: 会话ID（可选）
            inputs: 输入变量，默认{}
            files: 文件列表
            
        Returns:
            Generator[Dict[str, Any], None, None]: 响应事件生成器
        """
        # 处理文件参数
        sdk_files = None
        if files:
            sdk_files = []
            for file in files:
                if file["transfer_method"] == "local_file":
                    sdk_files.append({
                        "type": file["type"],
                        "transfer_method": "local_file",
                        "upload_file_id": file["upload_file_id"]
                    })
                elif file["transfer_method"] == "remote_url":
                    sdk_files.append({
                        "type": file["type"],
                        "transfer_method": "remote_url",
                        "url": file["url"]
                    })
        
        # 使用SDK调用流式聊天接口
        response = self.client.chat_completion(
            inputs=inputs or {},
            query=query,
            user=user,
            response_mode="streaming",
            conversation_id=conversation_id,
            files=sdk_files
        )
        
        # 返回生成器
        return response
    
    async def async_chat_completion(self, 
                                  query: str, 
                                  user: str = "default", 
                                  conversation_id: Optional[str] = None,
                                  inputs: Optional[Dict[str, Any]] = None,
                                  response_mode: str = "streaming",
                                  files: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        异步发送聊天消息
        
        Args:
            query: 用户输入/提问内容
            user: 用户标识
            conversation_id: 会话ID（可选）
            inputs: 输入变量，默认{}
            response_mode: 响应模式 (streaming或blocking)
            files: 文件列表
            
        Returns:
            Dict[str, Any]: 聊天响应结果
        """
        # 处理文件参数
        sdk_files = None
        if files:
            sdk_files = []
            for file in files:
                if file["transfer_method"] == "local_file":
                    sdk_files.append({
                        "type": file["type"],
                        "transfer_method": "local_file",
                        "upload_file_id": file["upload_file_id"]
                    })
                elif file["transfer_method"] == "remote_url":
                    sdk_files.append({
                        "type": file["type"],
                        "transfer_method": "remote_url",
                        "url": file["url"]
                    })
        
        # 使用SDK调用异步聊天接口
        response = await self.client.async_chat_completion(
            inputs=inputs or {},
            query=query,
            user=user,
            response_mode="blocking" if response_mode == "blocking" else "streaming",
            conversation_id=conversation_id,
            files=sdk_files
        )
        
        # 如果是阻塞模式直接返回结果
        if response_mode == "blocking":
            return response
        
        # 如果是流式模式，异步收集完整响应
        return await self._async_collect_streaming_response(response)
    
    async def async_streaming_chat_completion(self, 
                                           query: str, 
                                           user: str = "default", 
                                           conversation_id: Optional[str] = None,
                                           inputs: Optional[Dict[str, Any]] = None,
                                           files: Optional[List[Dict[str, Any]]] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        异步流式聊天消息（异步生成器模式）
        
        Args:
            query: 用户输入/提问内容
            user: 用户标识
            conversation_id: 会话ID（可选）
            inputs: 输入变量，默认{}
            files: 文件列表
            
        Returns:
            AsyncGenerator[Dict[str, Any], None]: 异步响应事件生成器
        """
        # 处理文件参数
        sdk_files = None
        if files:
            sdk_files = []
            for file in files:
                if file["transfer_method"] == "local_file":
                    sdk_files.append({
                        "type": file["type"],
                        "transfer_method": "local_file",
                        "upload_file_id": file["upload_file_id"]
                    })
                elif file["transfer_method"] == "remote_url":
                    sdk_files.append({
                        "type": file["type"],
                        "transfer_method": "remote_url",
                        "url": file["url"]
                    })
        
        # 使用SDK调用异步流式聊天接口
        response = await self.client.async_chat_completion(
            inputs=inputs or {},
            query=query,
            user=user,
            response_mode="streaming",
            conversation_id=conversation_id,
            files=sdk_files
        )
        
        # 返回异步生成器
        return response
    
    def upload_file(self, file_path: str, user: str = "default") -> Dict[str, Any]:
        """
        上传文件（图片）
        
        Args:
            file_path: 本地文件路径
            user: 用户标识
            
        Returns:
            Dict[str, Any]: 上传文件响应
            
        Raises:
            ValueError: 当文件不存在或格式不支持时
        """
        # 验证文件是否存在
        if not os.path.exists(file_path):
            raise ValueError(f"文件不存在: {file_path}")
            
        # 获取文件扩展名并验证格式
        file_extension = os.path.splitext(file_path)[1].lower().lstrip('.')
        supported_extensions = ['png', 'jpg', 'jpeg', 'webp', 'gif']
        
        if file_extension not in supported_extensions:
            raise ValueError(f"不支持的文件格式，仅支持: {', '.join(supported_extensions)}")
        
        # 使用SDK上传文件
        return self.client.file_upload(file_path=file_path, user=user)
    
    async def async_upload_file(self, file_path: str, user: str = "default") -> Dict[str, Any]:
        """
        异步上传文件
        
        Args:
            file_path: 本地文件路径
            user: 用户标识
            
        Returns:
            Dict[str, Any]: 上传文件响应
        """
        # 验证文件是否存在
        if not os.path.exists(file_path):
            raise ValueError(f"文件不存在: {file_path}")
            
        # 获取文件扩展名并验证格式
        file_extension = os.path.splitext(file_path)[1].lower().lstrip('.')
        supported_extensions = ['png', 'jpg', 'jpeg', 'webp', 'gif']
        
        if file_extension not in supported_extensions:
            raise ValueError(f"不支持的文件格式，仅支持: {', '.join(supported_extensions)}")
        
        # 使用SDK异步上传文件
        return await self.client.async_file_upload(file_path=file_path, user=user)
    
    def get_conversation_messages(self, 
                                conversation_id: str, 
                                user: str = "default", 
                                limit: int = 20,
                                last_id: Optional[str] = None) -> Dict[str, Any]:
        """
        获取对话消息历史
        
        Args:
            conversation_id: 对话ID
            user: 用户标识
            limit: 消息数量限制
            last_id: 上一页最后消息ID
            
        Returns:
            Dict[str, Any]: 对话消息历史
        """
        return self.client.get_conversation_messages(
            conversation_id=conversation_id,
            user=user,
            limit=limit,
            last_id=last_id
        )
    
    def get_conversations(self, 
                        user: str = "default", 
                        limit: int = 20,
                        first_id: Optional[str] = None,
                        last_id: Optional[str] = None) -> Dict[str, Any]:
        """
        获取对话列表
        
        Args:
            user: 用户标识
            limit: 对话数量限制
            first_id: 第一页第一个对话ID
            last_id: 最后一页最后对话ID
            
        Returns:
            Dict[str, Any]: 对话列表
        """
        return self.client.get_conversations(
            user=user,
            limit=limit,
            first_id=first_id,
            last_id=last_id
        )
    
    def rename_conversation(self, 
                          conversation_id: str, 
                          name: str, 
                          user: str = "default") -> Dict[str, Any]:
        """
        重命名对话
        
        Args:
            conversation_id: 对话ID
            name: 新名称
            user: 用户标识
            
        Returns:
            Dict[str, Any]: 重命名响应
        """
        return self.client.rename_conversation(
            conversation_id=conversation_id,
            name=name,
            user=user
        )
    
    def delete_conversation(self, 
                          conversation_id: str, 
                          user: str = "default") -> Dict[str, Any]:
        """
        删除对话
        
        Args:
            conversation_id: 对话ID
            user: 用户标识
            
        Returns:
            Dict[str, Any]: 删除响应
        """
        return self.client.delete_conversation(
            conversation_id=conversation_id,
            user=user
        )
    
    def stop_completion(self, task_id: str, user: str = "default") -> Dict[str, Any]:
        """
        停止生成
        
        Args:
            task_id: 任务ID
            user: 用户标识
            
        Returns:
            Dict[str, Any]: 停止响应
        """
        return self.client.stop_completion(task_id=task_id, user=user)
    
    def _collect_streaming_response(self, stream_generator) -> Dict[str, Any]:
        """
        收集流式响应
        
        Args:
            stream_generator: 流式响应生成器
            
        Returns:
            Dict[str, Any]: 完整响应
        """
        full_answer = ""
        metadata = None
        message_id = None
        conversation_id = None
        task_id = None
        
        for event in stream_generator:
            event_type = event.get("event")
            
            if event_type in ["message", "agent_message"]:
                full_answer += event.get("answer", "")
                # 记录关键信息
                if "message_id" in event and not message_id:
                    message_id = event.get("message_id")
                if "conversation_id" in event and not conversation_id:
                    conversation_id = event.get("conversation_id")
                if "task_id" in event and not task_id:
                    task_id = event.get("task_id")
                    
            elif event_type == "message_end":
                metadata = event.get("metadata")
        
        return {
            "answer": full_answer,
            "message_id": message_id,
            "conversation_id": conversation_id,
            "task_id": task_id,
            "metadata": metadata
        }
    
    async def _async_collect_streaming_response(self, stream_generator) -> Dict[str, Any]:
        """
        异步收集流式响应
        
        Args:
            stream_generator: 异步流式响应生成器
            
        Returns:
            Dict[str, Any]: 完整响应
        """
        full_answer = ""
        metadata = None
        message_id = None
        conversation_id = None
        task_id = None
        
        async for event in stream_generator:
            event_type = event.get("event")
            
            if event_type in ["message", "agent_message"]:
                full_answer += event.get("answer", "")
                # 记录关键信息
                if "message_id" in event and not message_id:
                    message_id = event.get("message_id")
                if "conversation_id" in event and not conversation_id:
                    conversation_id = event.get("conversation_id")
                if "task_id" in event and not task_id:
                    task_id = event.get("task_id")
                    
            elif event_type == "message_end":
                metadata = event.get("metadata")
        
        return {
            "answer": full_answer,
            "message_id": message_id,
            "conversation_id": conversation_id,
            "task_id": task_id,
            "metadata": metadata
        }

    # === 知识库操作接口 ===
    
    def get_dataset_list(self) -> Dict[str, Any]:
        """
        获取知识库列表
        
        Returns:
            Dict[str, Any]: 知识库列表
        """
        # 通过SDK的rest接口调用知识库列表API
        return self.client.rest.get(path="console/datasets")
    
    def get_dataset_detail(self, dataset_id: str) -> Dict[str, Any]:
        """
        获取知识库详情
        
        Args:
            dataset_id: 知识库ID
            
        Returns:
            Dict[str, Any]: 知识库详情
        """
        return self.client.rest.get(path=f"console/datasets/{dataset_id}")
    
    def create_dataset(self, name: str, description: str = "", 
                     indexing_technique: str = "high_quality") -> Dict[str, Any]:
        """
        创建知识库
        
        Args:
            name: 知识库名称
            description: 知识库描述
            indexing_technique: 索引技术，可选值: high_quality, economy
            
        Returns:
            Dict[str, Any]: 创建结果
        """
        data = {
            "name": name,
            "description": description,
            "indexing_technique": indexing_technique
        }
        return self.client.rest.post(path="console/datasets", json=data)
    
    def update_dataset(self, dataset_id: str, 
                     name: Optional[str] = None, 
                     description: Optional[str] = None) -> Dict[str, Any]:
        """
        更新知识库
        
        Args:
            dataset_id: 知识库ID
            name: 知识库名称
            description: 知识库描述
            
        Returns:
            Dict[str, Any]: 更新结果
        """
        data = {}
        if name is not None:
            data["name"] = name
        if description is not None:
            data["description"] = description
            
        return self.client.rest.patch(path=f"console/datasets/{dataset_id}", json=data)
    
    def delete_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """
        删除知识库
        
        Args:
            dataset_id: 知识库ID
            
        Returns:
            Dict[str, Any]: 删除结果
        """
        return self.client.rest.delete(path=f"console/datasets/{dataset_id}")
    
    def get_documents(self, dataset_id: str, page: int = 1, limit: int = 20) -> Dict[str, Any]:
        """
        获取知识库文档列表
        
        Args:
            dataset_id: 知识库ID
            page: 页码
            limit: 每页条数
            
        Returns:
            Dict[str, Any]: 文档列表
        """
        params = {
            "page": page,
            "limit": limit
        }
        return self.client.rest.get(path=f"console/datasets/{dataset_id}/documents", params=params)
    
    def upload_document(self, dataset_id: str, file_path: str) -> Dict[str, Any]:
        """
        上传文档到知识库
        
        Args:
            dataset_id: 知识库ID
            file_path: 本地文件路径
            
        Returns:
            Dict[str, Any]: 上传结果
        
        Raises:
            ValueError: 当文件不存在时
        """
        # 验证文件是否存在
        if not os.path.exists(file_path):
            raise ValueError(f"文件不存在: {file_path}")
        
        # 准备文件数据
        with open(file_path, 'rb') as file:
            files = {'file': (os.path.basename(file_path), file)}
            
            # 使用SDK的rest接口上传文件
            return self.client.rest.post_with_files(
                path=f"console/datasets/{dataset_id}/documents",
                files=files
            )
            
    async def async_upload_document(self, dataset_id: str, file_path: str) -> Dict[str, Any]:
        """
        异步上传文档到知识库
        
        Args:
            dataset_id: 知识库ID
            file_path: 本地文件路径
            
        Returns:
            Dict[str, Any]: 上传结果
            
        Raises:
            ValueError: 当文件不存在时
        """
        # 验证文件是否存在
        if not os.path.exists(file_path):
            raise ValueError(f"文件不存在: {file_path}")
        
        # 准备文件数据，使用线程池执行IO操作
        loop = asyncio.get_event_loop()
        
        def read_file():
            with open(file_path, 'rb') as file:
                return file.read()
                
        file_content = await loop.run_in_executor(None, read_file)
        file_name = os.path.basename(file_path)
        
        # 使用SDK的rest接口异步上传文件
        return await self.client.rest.async_post_with_files(
            path=f"console/datasets/{dataset_id}/documents",
            files={'file': (file_name, file_content)}
        )
    
    def delete_document(self, dataset_id: str, document_id: str) -> Dict[str, Any]:
        """
        删除知识库文档
        
        Args:
            dataset_id: 知识库ID
            document_id: 文档ID
            
        Returns:
            Dict[str, Any]: 删除结果
        """
        return self.client.rest.delete(
            path=f"console/datasets/{dataset_id}/documents/{document_id}"
        )
    
    def segment_detail(self, dataset_id: str, segment_id: str) -> Dict[str, Any]:
        """
        获取文档分段详情
        
        Args:
            dataset_id: 知识库ID
            segment_id: 分段ID
            
        Returns:
            Dict[str, Any]: 分段详情
        """
        return self.client.rest.get(
            path=f"console/datasets/{dataset_id}/segment/{segment_id}"
        )
    
    def search_segments(self, dataset_id: str, query: str, page: int = 1, 
                       limit: int = 20, search_method: str = "semantic") -> Dict[str, Any]:
        """
        搜索知识库分段
        
        Args:
            dataset_id: 知识库ID
            query: 搜索关键词
            page: 页码
            limit: 每页条数
            search_method: 搜索方法，可选值: semantic, fulltext, hybrid
            
        Returns:
            Dict[str, Any]: 搜索结果
        """
        params = {
            "query": query,
            "page": page,
            "limit": limit,
            "search_method": search_method
        }
        return self.client.rest.get(
            path=f"console/datasets/{dataset_id}/segments",
            params=params
        )
    
    def process_dataset_index(self, dataset_id: str, action: str) -> Dict[str, Any]:
        """
        处理知识库索引（刷新、暂停或恢复）
        
        Args:
            dataset_id: 知识库ID
            action: 操作类型，可选值: refresh, pause, resume
            
        Returns:
            Dict[str, Any]: 操作结果
        """
        data = {"action": action}
        return self.client.rest.post(
            path=f"console/datasets/{dataset_id}/index/process",
            json=data
        )
        
    def get_dataset_index_status(self, dataset_id: str) -> Dict[str, Any]:
        """
        获取知识库索引状态
        
        Args:
            dataset_id: 知识库ID
            
        Returns:
            Dict[str, Any]: 索引状态
        """
        return self.client.rest.get(
            path=f"console/datasets/{dataset_id}/index/status"
        )
        
    def create_dataset_metadata(self, dataset_id: str, metadata_type: str, metadata_name: str) -> Dict[str, Any]:
        """
        新增知识库元数据
        
        Args:
            dataset_id: 知识库ID
            metadata_type: 元数据类型
            metadata_name: 元数据名称
            
        Returns:
            Dict[str, Any]: 操作结果
        """
        data = {
            "type": metadata_type,
            "name": metadata_name
        }
        
        return self.client.rest.post(
            path=f"v1/datasets/{dataset_id}/metadata",
            json=data
        )
        
    def update_dataset_metadata(self, dataset_id: str, metadata_id: str, metadata_name: str) -> Dict[str, Any]:
        """
        更新知识库元数据
        
        Args:
            dataset_id: 知识库ID
            metadata_id: 元数据ID
            metadata_name: 元数据名称
            
        Returns:
            Dict[str, Any]: 操作结果，包含：
                - id: 元数据ID
                - type: 元数据类型
                - name: 元数据名称
        """
        data = {
            "name": metadata_name
        }
        
        return self.client.rest.patch(
            path=f"v1/datasets/{dataset_id}/metadata/{metadata_id}",
            json=data
        )
        
    def get_dataset_metadata(self, dataset_id: str) -> Dict[str, Any]:
        """
        获取知识库元数据列表
        
        Args:
            dataset_id: 知识库ID
            
        Returns:
            Dict[str, Any]: 元数据列表
        """
        return self.client.rest.get(
            path=f"v1/datasets/{dataset_id}/metadata"
        )
        
    def delete_dataset_metadata(self, dataset_id: str, metadata_id: str) -> Dict[str, Any]:
        """
        删除知识库元数据
        
        Args:
            dataset_id: 知识库ID
            metadata_id: 元数据ID
            
        Returns:
            Dict[str, Any]: 操作结果
        """
        return self.client.rest.delete(
            path=f"v1/datasets/{dataset_id}/metadata/{metadata_id}"
        )
