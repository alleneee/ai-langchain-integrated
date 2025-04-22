"""
Dify API 客户端模块

该模块提供了与Dify平台API交互的客户端实现
"""

import json
import logging
import requests
import sseclient
from typing import Dict, Any, List, Optional, Union, Generator, Callable
from urllib.parse import urljoin
import base64
import os

logger = logging.getLogger(__name__)

class DifyStreamResponse:
    """
    Dify 流式响应处理类
    用于处理SSE流式响应并解析各种事件类型
    """
    
    def __init__(self, response):
        """
        初始化流式响应处理器
        
        Args:
            response: SSE响应对象
        """
        self.client = sseclient.SSEClient(response)
        self.task_id = None
        self.conversation_id = None
        self.message_id = None
        
    def __iter__(self):
        """迭代器方法，允许直接迭代事件"""
        return self
        
    def __next__(self):
        """获取下一个事件"""
        event = next(self.client)
        if event.data:
            try:
                data = json.loads(event.data)
                # 保存重要的标识符供后续使用
                if 'task_id' in data:
                    self.task_id = data['task_id']
                if 'conversation_id' in data:
                    self.conversation_id = data['conversation_id']
                if 'message_id' in data and data.get('event') != 'error':
                    self.message_id = data['message_id']
                return data
            except json.JSONDecodeError:
                logger.error(f"无法解析事件数据: {event.data}")
                return {"event": "error", "message": "数据解析错误"}
        return {"event": "empty", "message": "空数据"}
    
    def events(self) -> Generator[Dict[str, Any], None, None]:
        """
        生成器方法，返回所有事件
        
        Returns:
            Generator[Dict[str, Any], None, None]: 事件生成器
        """
        for event in self:
            yield event

    def collect_full_response(self) -> Dict[str, Any]:
        """
        收集完整的响应内容
        
        Returns:
            Dict[str, Any]: 完整的响应内容，包括全文和元数据
        """
        full_answer = ""
        metadata = None
        tts_audio_parts = []
        agent_thoughts = []
        files = []
        error = None
        
        for event in self:
            event_type = event.get('event')
            
            if event_type in ['message', 'agent_message']:
                full_answer += event.get('answer', '')
            elif event_type == 'message_end':
                metadata = event.get('metadata')
            elif event_type == 'tts_message':
                if event.get('audio'):
                    tts_audio_parts.append(event.get('audio'))
            elif event_type == 'agent_thought':
                agent_thoughts.append(event)
            elif event_type == 'message_file':
                files.append(event)
            elif event_type == 'error':
                error = event
                break
        
        # 合并TTS音频数据
        tts_audio = None
        if tts_audio_parts:
            tts_audio = ''.join(tts_audio_parts)
            
        return {
            'answer': full_answer,
            'conversation_id': self.conversation_id,
            'message_id': self.message_id,
            'task_id': self.task_id,
            'metadata': metadata,
            'tts_audio': tts_audio,
            'agent_thoughts': agent_thoughts,
            'files': files,
            'error': error
        }
        
    def process_with_callbacks(self, 
                             on_message: Optional[Callable[[str], None]] = None,
                             on_end: Optional[Callable[[Dict[str, Any]], None]] = None,
                             on_error: Optional[Callable[[Dict[str, Any]], None]] = None,
                             on_agent_thought: Optional[Callable[[Dict[str, Any]], None]] = None,
                             on_tts: Optional[Callable[[str], None]] = None,
                             on_file: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        """
        使用回调函数处理流式响应
        
        Args:
            on_message: 收到文本内容时的回调
            on_end: 消息结束时的回调
            on_error: 出现错误时的回调
            on_agent_thought: 收到Agent思考内容时的回调
            on_tts: 收到TTS音频时的回调
            on_file: 收到文件时的回调
            
        Returns:
            Dict[str, Any]: 完整的响应内容
        """
        full_answer = ""
        metadata = None
        
        for event in self:
            event_type = event.get('event')
            
            if event_type == 'message':
                chunk = event.get('answer', '')
                full_answer += chunk
                if on_message:
                    on_message(chunk)
                    
            elif event_type == 'agent_message':
                chunk = event.get('answer', '')
                full_answer += chunk
                if on_message:
                    on_message(chunk)
                    
            elif event_type == 'message_end':
                metadata = event.get('metadata')
                if on_end:
                    on_end(event)
                    
            elif event_type == 'tts_message':
                if on_tts and event.get('audio'):
                    on_tts(event.get('audio'))
                    
            elif event_type == 'agent_thought':
                if on_agent_thought:
                    on_agent_thought(event)
                    
            elif event_type == 'message_file':
                if on_file:
                    on_file(event)
                    
            elif event_type == 'error':
                if on_error:
                    on_error(event)
                break
        
        return {
            'answer': full_answer,
            'conversation_id': self.conversation_id,
            'message_id': self.message_id,
            'task_id': self.task_id,
            'metadata': metadata
        }

class DifyAPIClient:
    """
    Dify API 客户端类，用于与Dify平台的API进行交互
    """
    
    def __init__(self, api_base_url: str, api_key: str, timeout: int = 60):
        """
        初始化Dify API客户端
        
        Args:
            api_base_url: Dify API基础URL
            api_key: Dify API密钥
            timeout: 请求超时时间（秒）
        """
        self.api_base_url = api_base_url.rstrip("/") + "/"
        self.api_key = api_key
        self.timeout = timeout
    
    def _get_headers(self, is_app_api: bool = False) -> Dict[str, str]:
        """
        获取请求头
        
        Args:
            is_app_api: 是否为App API（True）或者Console API（False）
            
        Returns:
            Dict[str, str]: 请求头字典
        """
        if is_app_api:
            return {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        else:
            return {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None, 
                     params: Optional[Dict[str, Any]] = None, is_app_api: bool = False,
                     files: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        发送HTTP请求到Dify API
        
        Args:
            method: HTTP方法（GET, POST, PUT, DELETE等）
            endpoint: API端点路径
            data: 请求体数据
            params: URL查询参数
            is_app_api: 是否为App API
            files: 文件上传数据
            
        Returns:
            Dict[str, Any]: API响应
            
        Raises:
            Exception: 当API请求失败时
        """
        url = urljoin(self.api_base_url, endpoint)
        headers = self._get_headers(is_app_api)
        
        try:
            if files:
                # 文件上传请求不设置Content-Type，由requests自动设置
                headers.pop("Content-Type", None)
            
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                json=data if data and not files else None,
                params=params,
                files=files,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Dify API请求失败: {str(e)}")
            try:
                error_detail = response.json() if response.content else {}
                logger.error(f"API错误响应: {error_detail}")
            except:
                error_detail = {"error": str(e)}
            
            raise Exception(f"Dify API请求失败: {str(e)}")
    
    def _make_streaming_request(self, endpoint: str, data: Dict[str, Any], 
                              is_app_api: bool = True) -> DifyStreamResponse:
        """
        发送流式请求到Dify API
        
        Args:
            endpoint: API端点路径
            data: 请求体数据
            is_app_api: 是否为App API
            
        Returns:
            DifyStreamResponse: 流式响应处理器
            
        Raises:
            Exception: 当API请求失败时
        """
        url = urljoin(self.api_base_url, endpoint)
        headers = self._get_headers(is_app_api)
        
        try:
            response = requests.post(
                url=url,
                headers=headers,
                json=data,
                stream=True,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            return DifyStreamResponse(response)
        except requests.exceptions.RequestException as e:
            logger.error(f"Dify API流式请求失败: {str(e)}")
            try:
                error_detail = response.json() if response.content else {}
                logger.error(f"API错误响应: {error_detail}")
            except:
                error_detail = {"error": str(e)}
            
            raise Exception(f"Dify API流式请求失败: {str(e)}")

    # === App API - 对话接口 ===
    
    def chat_messages(self, app_id: str, query: str, inputs: Optional[Dict[str, Any]] = None,
                    conversation_id: Optional[str] = None, user: str = "default",
                    response_mode: str = "streaming", files: Optional[List[Dict[str, Any]]] = None,
                    auto_generate_name: bool = True) -> Dict[str, Any]:
        """
        发送聊天消息
        
        Args:
            app_id: 应用ID
            query: 用户输入/提问内容
            inputs: 允许传入App定义的各变量值，默认{}
            conversation_id: 会话ID（可选，不提供则创建新对话）
            user: 用户标识，用于定义终端用户的身份，需保证在应用内唯一
            response_mode: 响应模式
                - streaming: 流式模式（推荐），基于SSE实现类似打字机输出
                - blocking: 阻塞模式，等待执行完毕后返回结果（请求较长可能被中断）
            files: 上传的文件列表，每个文件为一个字典，包含：
                - type: 文件类型，目前支持'image'
                - transfer_method: 传递方式，'remote_url'或'local_file'
                - url: 图片地址（当transfer_method为'remote_url'时）
                - upload_file_id: 上传文件ID（当transfer_method为'local_file'时）
            auto_generate_name: 是否自动生成标题，默认True
            
        Returns:
            Dict[str, Any]: 聊天消息响应
        """
        data = {
            "inputs": inputs or {},
            "query": query,
            "response_mode": response_mode,
            "user": user,
            "auto_generate_name": auto_generate_name
        }
        
        if conversation_id:
            data["conversation_id"] = conversation_id
            
        if files:
            data["files"] = files
        
        # 阻塞模式使用普通请求
        if response_mode == "blocking":
            return self._make_request(
                method="POST",
                endpoint=f"v1/chat-messages",
                data=data,
                is_app_api=True
            )
        # 流式模式使用流式请求
        else:
            return self._make_streaming_request(
                endpoint=f"v1/chat-messages",
                data=data,
                is_app_api=True
            ).collect_full_response()
    
    def chat_messages_stream(self, app_id: str, query: str, inputs: Optional[Dict[str, Any]] = None,
                           conversation_id: Optional[str] = None, user: str = "default",
                           files: Optional[List[Dict[str, Any]]] = None,
                           auto_generate_name: bool = True) -> DifyStreamResponse:
        """
        发送流式聊天消息（直接返回流式响应对象）
        
        Args:
            app_id: 应用ID
            query: 用户输入/提问内容
            inputs: 允许传入App定义的各变量值，默认{}
            conversation_id: 会话ID（可选，不提供则创建新对话）
            user: 用户标识，用于定义终端用户的身份，需保证在应用内唯一
            files: 上传的文件列表，每个文件为一个字典，包含：
                - type: 文件类型，目前支持'image'
                - transfer_method: 传递方式，'remote_url'或'local_file'
                - url: 图片地址（当transfer_method为'remote_url'时）
                - upload_file_id: 上传文件ID（当transfer_method为'local_file'时）
            auto_generate_name: 是否自动生成标题，默认True
            
        Returns:
            DifyStreamResponse: 流式响应处理器，可用于迭代处理事件
        """
        data = {
            "inputs": inputs or {},
            "query": query,
            "response_mode": "streaming",
            "user": user,
            "auto_generate_name": auto_generate_name
        }
        
        if conversation_id:
            data["conversation_id"] = conversation_id
            
        if files:
            data["files"] = files
        
        return self._make_streaming_request(
            endpoint=f"v1/chat-messages",
            data=data,
            is_app_api=True
        )
        
    def stop_chat_response(self, task_id: str) -> Dict[str, Any]:
        """
        停止流式响应
        
        Args:
            task_id: 任务ID
            
        Returns:
            Dict[str, Any]: 停止响应的结果
        """
        return self._make_request(
            method="POST",
            endpoint=f"v1/stop-generate",
            data={"task_id": task_id},
            is_app_api=True
        )

    # === 文件处理 ===
    
    def upload_file(self, file_path: str, user: str = "default", mime_type: Optional[str] = None) -> Dict[str, Any]:
        """
        上传文件（目前仅支持图片）
        
        支持的图片格式：png, jpg, jpeg, webp, gif
        上传的文件仅供当前终端用户使用，必须与发送消息时的user保持一致
        
        Args:
            file_path: 本地文件路径
            user: 用户标识，必须与发送消息接口传入user保持一致
            mime_type: 文件MIME类型，如不提供则根据文件扩展名自动判断
            
        Returns:
            Dict[str, Any]: 文件上传响应，包含：
                - id: 文件唯一ID
                - name: 文件名
                - size: 文件大小(byte)
                - extension: 文件后缀
                - mime_type: 文件MIME类型
                - created_by: 上传人ID
                - created_at: 上传时间戳
                
        Raises:
            ValueError: 当文件不存在或格式不支持时
            Exception: 当API请求失败时
        """
        # 验证文件是否存在
        if not os.path.exists(file_path):
            raise ValueError(f"文件不存在: {file_path}")
            
        # 获取文件扩展名并验证格式
        file_extension = os.path.splitext(file_path)[1].lower().lstrip('.')
        supported_extensions = ['png', 'jpg', 'jpeg', 'webp', 'gif']
        
        if file_extension not in supported_extensions:
            raise ValueError(f"不支持的文件格式，仅支持: {', '.join(supported_extensions)}")
        
        # 如果未提供MIME类型，则根据扩展名确定
        if not mime_type:
            mime_types = {
                'png': 'image/png',
                'jpg': 'image/jpeg',
                'jpeg': 'image/jpeg',
                'webp': 'image/webp',
                'gif': 'image/gif'
            }
            mime_type = mime_types.get(file_extension, f'image/{file_extension}')
        
        # 准备表单数据
        with open(file_path, 'rb') as file:
            file_name = os.path.basename(file_path)
            files = {
                'file': (file_name, file, mime_type)
            }
            data = {
                'user': user
            }
            
            # 上传文件
            url = urljoin(self.api_base_url, "v1/files/upload")
            headers = self._get_headers(is_app_api=True)
            # multipart/form-data请求不应指定Content-Type，由requests库自动设置
            if 'Content-Type' in headers:
                headers.pop('Content-Type')
                
            try:
                response = requests.post(
                    url=url,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.error(f"文件上传失败: {str(e)}")
                try:
                    error_detail = response.json() if response.content else {}
                    logger.error(f"API错误响应: {error_detail}")
                except:
                    error_detail = {"error": str(e)}
                
                raise Exception(f"文件上传失败: {str(e)}")
    
    def get_file_info(self, file_id: str) -> Dict[str, Any]:
        """
        获取文件信息
        
        Args:
            file_id: 文件ID
            
        Returns:
            Dict[str, Any]: 文件信息
        """
        return self._make_request(
            method="GET",
            endpoint=f"v1/files/{file_id}",
            is_app_api=True
        )
        
    # === Console API - 应用管理 ===
    
    def get_applications(self, page: int = 1, limit: int = 20) -> Dict[str, Any]:
        """
        获取应用列表
        
        Args:
            page: 页码
            limit: 每页数量
            
        Returns:
            Dict[str, Any]: 应用列表响应
        """
        return self._make_request(
            method="GET",
            endpoint="console/apps",
            params={"page": page, "limit": limit}
        )
    
    def get_application(self, app_id: str) -> Dict[str, Any]:
        """
        获取应用详情
        
        Args:
            app_id: 应用ID
            
        Returns:
            Dict[str, Any]: 应用详情响应
        """
        return self._make_request(
            method="GET",
            endpoint=f"console/apps/{app_id}"
        )
    
    def create_application(self, name: str, description: str, 
                         mode: str = "chat", icon: str = "emoji", 
                         icon_background: str = "#FFEAD5") -> Dict[str, Any]:
        """
        创建应用
        
        Args:
            name: 应用名称
            description: 应用描述
            mode: 应用模式（chat或completion）
            icon: 应用图标
            icon_background: 图标背景色
            
        Returns:
            Dict[str, Any]: 创建应用响应
        """
        data = {
            "name": name,
            "description": description,
            "mode": mode,
            "icon": icon,
            "icon_background": icon_background
        }
        
        return self._make_request(
            method="POST",
            endpoint="console/apps",
            data=data
        )
    
    def update_application(self, app_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        更新应用
        
        Args:
            app_id: 应用ID
            data: 更新数据
            
        Returns:
            Dict[str, Any]: 更新应用响应
        """
        return self._make_request(
            method="PATCH",
            endpoint=f"console/apps/{app_id}",
            data=data
        )
    
    def delete_application(self, app_id: str) -> Dict[str, Any]:
        """
        删除应用
        
        Args:
            app_id: 应用ID
            
        Returns:
            Dict[str, Any]: 删除应用响应
        """
        return self._make_request(
            method="DELETE",
            endpoint=f"console/apps/{app_id}"
        )
        
    # === 对话管理接口 ===
    
    def get_conversation_messages(self, app_id: str, conversation_id: str, 
                                user: str = "default", limit: int = 20, 
                                last_id: Optional[str] = None) -> Dict[str, Any]:
        """
        获取对话消息历史
        
        Args:
            app_id: 应用ID
            conversation_id: 对话ID
            user: 用户标识
            limit: 消息数量限制
            last_id: 上一页最后消息ID
            
        Returns:
            Dict[str, Any]: 对话消息历史
        """
        params = {
            "conversation_id": conversation_id,
            "user": user,
            "limit": limit
        }
        
        if last_id:
            params["last_id"] = last_id
        
        return self._make_request(
            method="GET",
            endpoint="v1/messages",
            params=params,
            is_app_api=True
        )
    
    def get_conversations(self, app_id: str, user: str = "default", 
                        limit: int = 20, first_id: Optional[str] = None,
                        last_id: Optional[str] = None) -> Dict[str, Any]:
        """
        获取对话列表
        
        Args:
            app_id: 应用ID
            user: 用户标识
            limit: 对话数量限制
            first_id: 第一页第一个对话ID
            last_id: 最后一页最后对话ID
            
        Returns:
            Dict[str, Any]: 对话列表
        """
        params = {
            "user": user,
            "limit": limit
        }
        
        if first_id:
            params["first_id"] = first_id
            
        if last_id:
            params["last_id"] = last_id
        
        return self._make_request(
            method="GET",
            endpoint="v1/conversations",
            params=params,
            is_app_api=True
        )
    
    def delete_conversation(self, app_id: str, conversation_id: str, 
                          user: str = "default") -> Dict[str, Any]:
        """
        删除对话
        
        Args:
            app_id: 应用ID
            conversation_id: 对话ID
            user: 用户标识
            
        Returns:
            Dict[str, Any]: 删除响应
        """
        return self._make_request(
            method="DELETE",
            endpoint=f"v1/conversations/{conversation_id}",
            params={"user": user},
            is_app_api=True
        )
    
    def rename_conversation(self, app_id: str, conversation_id: str, 
                          name: str, user: str = "default") -> Dict[str, Any]:
        """
        重命名对话
        
        Args:
            app_id: 应用ID
            conversation_id: 对话ID
            name: 新名称
            user: 用户标识
            
        Returns:
            Dict[str, Any]: 重命名响应
        """
        data = {
            "name": name,
            "user": user
        }
        
        return self._make_request(
            method="PATCH",
            endpoint=f"v1/conversations/{conversation_id}",
            data=data,
            is_app_api=True
        )
