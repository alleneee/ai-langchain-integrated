"""
聊天模式模块

该模块定义了聊天相关的请求和响应模式。
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from src.schemas.base import UserBase

# 聊天请求模式
class ChatMessageRequest(UserBase):
    """聊天消息请求"""
    inputs: Dict[str, Any] = Field({}, description="输入参数字典")
    query: str = Field(..., description="查询文本")
    response_mode: str = Field("blocking", description="响应模式：blocking或streaming")
    conversation_id: Optional[str] = Field(None, description="会话ID，不提供则创建新会话")
    files: Optional[List[str]] = Field(None, description="文件ID列表")

class ConversationListRequest(UserBase):
    """会话列表请求"""
    last_id: Optional[str] = Field(None, description="上一页最后一条记录ID")
    limit: Optional[int] = Field(None, description="每页记录数")
    pinned: Optional[bool] = Field(None, description="是否只显示已固定的会话")

class ConversationMessagesRequest(UserBase):
    """会话消息请求"""
    conversation_id: Optional[str] = Field(None, description="会话ID")
    first_id: Optional[str] = Field(None, description="第一条消息ID")
    limit: Optional[int] = Field(None, description="每页记录数")

class RenameConversationRequest(UserBase):
    """重命名会话请求"""
    conversation_id: str = Field(..., description="会话ID")
    name: str = Field(..., description="新名称")
    auto_generate: bool = Field(False, description="是否自动生成名称")

class DeleteConversationRequest(UserBase):
    """删除会话请求"""
    conversation_id: str = Field(..., description="会话ID")

class StopMessageRequest(UserBase):
    """停止生成消息请求"""
    task_id: str = Field(..., description="消息任务ID")

class MessageFeedbackRequest(UserBase):
    """消息反馈请求"""
    message_id: str = Field(..., description="消息ID")
    rating: str = Field(..., description="评分，如：like, dislike等")

class TextToAudioRequest(UserBase):
    """文本转语音请求"""
    text: str = Field(..., description="要转换的文本")
    streaming: bool = Field(False, description="是否流式传输")

class AudioToTextRequest(UserBase):
    """语音转文本请求"""
    # 注意：此模型在实际使用时可能需要处理文件上传

# 聊天响应模式
class Message(BaseModel):
    """消息模型"""
    id: str = Field(..., description="消息ID")
    conversation_id: str = Field(..., description="会话ID")
    role: str = Field(..., description="消息角色，如user或assistant")
    content: str = Field(..., description="消息内容")
    created_at: str = Field(..., description="创建时间")

class Conversation(BaseModel):
    """会话模型"""
    id: str = Field(..., description="会话ID")
    name: str = Field(..., description="会话名称")
    created_at: str = Field(..., description="创建时间")
    updated_at: str = Field(..., description="更新时间")
    pinned: bool = Field(False, description="是否置顶")
    
class ChatMessageResponse(BaseModel):
    """聊天消息响应"""
    id: str = Field(..., description="消息ID")
    conversation_id: str = Field(..., description="会话ID")
    answer: str = Field(..., description="回答内容")
    created_at: str = Field(..., description="创建时间") 