"""
模式包

该包含所有API请求和响应的数据模式。
"""

from src.schemas.base import UserBase, PaginationParams, OrderingParams
from src.schemas.responses import StandardResponse, DataResponse, PaginatedResponse
from src.schemas.chat import (
    ChatMessageRequest, ConversationListRequest, ConversationMessagesRequest,
    RenameConversationRequest, DeleteConversationRequest, StopMessageRequest,
    MessageFeedbackRequest, TextToAudioRequest, AudioToTextRequest,
    Message, Conversation, ChatMessageResponse
)
from src.schemas.completion import CompletionRequest, CompletionResponse
from src.schemas.workflow import WorkflowRunRequest, WorkflowStopRequest, WorkflowResult
from src.schemas.knowledge import (
    CreateDatasetRequest, DocumentListRequest, SegmentModel, AddSegmentsRequest,
    QuerySegmentsRequest, UpdateSegmentRequest, DeleteSegmentRequest,
    Dataset, Document, Segment
)
from src.schemas.chatbot import (
    ChatbotMessageRequest, ChatbotCreateRequest,
    ChatbotMessageResponse, ChatHistoryResponse
)
from src.schemas.llm import TokenEstimateRequest, TokenEstimateResponse, LLMProviderInfo
