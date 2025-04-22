"""
聊天机器人模式模块

该模块定义了聊天机器人相关的请求和响应模式。
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

class ChatbotMessageRequest(BaseModel):
    """聊天机器人请求模型"""
    provider: str = Field(..., description="提供商名称，支持 'openai', 'deepseek', 'qwen'")
    model_name: str = Field(..., description="模型名称")
    message: str = Field(..., description="用户消息")
    stream: bool = Field(False, description="是否使用流式返回")
    temperature: float = Field(0.7, description="温度参数，控制响应的随机性")
    max_tokens: Optional[int] = Field(None, description="生成文本的最大token数")
    system_message: Optional[str] = Field("你是一个有帮助的AI助手。", description="系统消息")
    
    # 提供商特定参数
    api_params: Dict[str, Any] = Field(default_factory=dict, description="API特定参数")

class ChatbotCreateRequest(BaseModel):
    """创建聊天机器人请求模型"""
    provider: str = Field(..., description="提供商名称，支持 'openai', 'deepseek', 'qwen'")
    model_name: str = Field(..., description="模型名称")
    streaming: bool = Field(False, description="是否使用流式返回")
    temperature: float = Field(0.7, description="温度参数，控制响应的随机性")
    max_tokens: Optional[int] = Field(None, description="生成文本的最大token数")
    system_message: Optional[str] = Field("你是一个有帮助的AI助手。", description="系统消息")
    
    # 提供商特定参数
    api_params: Dict[str, Any] = Field(default_factory=dict, description="API特定参数")

class ChatbotMessageResponse(BaseModel):
    """聊天机器人消息响应"""
    message: str = Field(..., description="AI助手回复")
    provider: str = Field(..., description="使用的提供商")
    model: str = Field(..., description="使用的模型")

class ChatHistoryResponse(BaseModel):
    """聊天历史响应"""
    history: List[Dict[str, Any]] = Field(..., description="聊天历史记录")
    provider: str = Field(..., description="使用的提供商")
    model: str = Field(..., description="使用的模型") 