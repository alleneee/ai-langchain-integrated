"""
聊天API接口模块

该模块实现了聊天相关的API接口。
"""

from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, Request
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, Optional
from src.services.chat_service import ChatService
from src.schemas.chat import (
    ChatMessageRequest, ConversationListRequest, ConversationMessagesRequest,
    RenameConversationRequest, DeleteConversationRequest, StopMessageRequest,
    MessageFeedbackRequest, TextToAudioRequest, AudioToTextRequest,
    Message, Conversation, ChatMessageResponse
)
from src.schemas.responses import DataResponse, StandardResponse
import json
import asyncio

# 创建路由器
router = APIRouter()

# 服务实例
chat_service = ChatService()

@router.on_event("startup")
async def startup_event():
    """启动事件处理函数"""
    await chat_service.initialize()

@router.post("/messages", response_model=DataResponse[ChatMessageResponse])
async def create_chat_message(request: ChatMessageRequest):
    """
    创建聊天消息
    
    Args:
        request: 聊天消息请求
        
    Returns:
        创建的消息结果
    """
    try:
        result = await chat_service.create_chat_message(
            inputs=request.inputs,
            query=request.query,
            user=request.user,
            response_mode=request.response_mode,
            conversation_id=request.conversation_id,
            files=request.files
        )
        return DataResponse(data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

@router.post("/messages/streaming")
async def create_streaming_chat_message(request: ChatMessageRequest):
    """
    创建流式聊天消息
    
    Args:
        request: 聊天消息请求
        
    Returns:
        流式聊天响应
    """
    async def stream_response():
        try:
            # 使用流式模式标记
            request.response_mode = "streaming"
            
            # 获取流式响应
            async for chunk in chat_service.create_streaming_chat_message(
                inputs=request.inputs,
                query=request.query,
                user=request.user,
                conversation_id=request.conversation_id,
                files=request.files
            ):
                # 将响应块转换为JSON字符串并添加行分隔符
                yield f"data: {json.dumps(chunk)}\n\n"
                
            # 发送结束标记
            yield "data: [DONE]\n\n"
        except Exception as e:
            error_data = {"error": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "X-Accel-Buffering": "no" # 用于Nginx代理时禁用缓冲
        }
    )

@router.get("/conversations", response_model=DataResponse[List[Conversation]])
async def get_conversations(request: ConversationListRequest):
    """
    获取会话列表
    
    Args:
        request: 会话列表请求
        
    Returns:
        会话列表
    """
    try:
        conversations = await chat_service.get_conversations(
            user=request.user,
            last_id=request.last_id,
            limit=request.limit,
            pinned=request.pinned
        )
        return DataResponse(data=conversations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

@router.get("/messages", response_model=DataResponse[List[Message]])
async def get_conversation_messages(request: ConversationMessagesRequest):
    """
    获取会话消息
    
    Args:
        request: 会话消息请求
        
    Returns:
        会话消息列表
    """
    try:
        messages = await chat_service.get_conversation_messages(
            user=request.user,
            conversation_id=request.conversation_id,
            first_id=request.first_id,
            limit=request.limit
        )
        return DataResponse(data=messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

@router.post("/conversations/rename", response_model=StandardResponse)
async def rename_conversation(request: RenameConversationRequest):
    """
    重命名会话
    
    Args:
        request: 重命名会话请求
        
    Returns:
        重命名结果
    """
    try:
        result = await chat_service.rename_conversation(
            conversation_id=request.conversation_id,
            name=request.name,
            auto_generate=request.auto_generate,
            user=request.user
        )
        return StandardResponse(status="success", message="会话重命名成功")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

@router.post("/conversations/delete", response_model=StandardResponse)
async def delete_conversation(request: DeleteConversationRequest):
    """
    删除会话
    
    Args:
        request: 删除会话请求
        
    Returns:
        删除结果
    """
    try:
        result = await chat_service.delete_conversation(
            conversation_id=request.conversation_id,
            user=request.user
        )
        return StandardResponse(status="success", message="会话删除成功")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

@router.post("/messages/stop", response_model=StandardResponse)
async def stop_message(request: StopMessageRequest):
    """
    停止消息生成
    
    Args:
        request: 停止消息请求
        
    Returns:
        停止结果
    """
    try:
        result = await chat_service.stop_message(
            task_id=request.task_id,
            user=request.user
        )
        return StandardResponse(status="success", message="消息生成已停止")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

@router.post("/messages/feedback", response_model=StandardResponse)
async def message_feedback(request: MessageFeedbackRequest):
    """
    提交消息反馈
    
    Args:
        request: 消息反馈请求
        
    Returns:
        反馈结果
    """
    try:
        result = await chat_service.message_feedback(
            message_id=request.message_id,
            rating=request.rating,
            user=request.user
        )
        return StandardResponse(status="success", message="反馈提交成功")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

@router.post("/text-to-audio", response_model=StandardResponse)
async def text_to_audio(request: TextToAudioRequest):
    """
    文本转语音
    
    Args:
        request: 文本转语音请求
        
    Returns:
        文本转语音结果
    """
    try:
        result = await chat_service.text_to_audio(
            text=request.text,
            user=request.user,
            streaming=request.streaming
        )
        return StandardResponse(status="success", message="文本转语音成功")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

@router.post("/audio-to-text")
async def audio_to_text(file: UploadFile = File(...), user: str = ""):
    """
    语音转文本
    
    Args:
        file: 语音文件
        user: 用户标识
        
    Returns:
        语音转文本结果
    """
    try:
        result = await chat_service.audio_to_text(
            audio_file=file.file,
            user=user
        )
        return DataResponse(data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")