"""
应用配置模块

该模块包含了应用程序的各种配置参数，采用分层结构组织所有配置。
"""

import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings

# 确定项目根目录
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()

# 定义子配置模型

class PlaywrightSettings(BaseModel):
    """Playwright MCP 设置"""
    enabled: bool = Field(
        default=True,
        description="是否启用 Playwright MCP"
    )
    headless: bool = Field(
        default=True,
        description="是否以无头模式运行浏览器"
    )
    port: int = Field(
        default=8080,
        description="MCP 服务器端口"
    )
    snapshot_mode: bool = Field(
        default=True,
        description="是否使用快照模式（如果为False则使用视觉模式）"
    )
    browser: str = Field(
        default="chromium",
        description="使用的浏览器类型 (chromium, firefox, webkit)"
    )
    timeout: int = Field(
        default=30000,
        description="操作超时时间（毫秒）"
    )
    viewport_width: int = Field(
        default=1280,
        description="浏览器视口宽度"
    )
    viewport_height: int = Field(
        default=720,
        description="浏览器视口高度"
    )
    auto_close: bool = Field(
        default=True,
        description="应用程序关闭时是否自动关闭 MCP 服务器"
    )

class BraveSearchSettings(BaseModel):
    """Brave Search MCP 设置"""
    enabled: bool = Field(
        default=True,
        description="是否启用 Brave Search MCP"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Brave Search API 密钥"
    )
    country: str = Field(
        default="CN",
        description="搜索国家或地区代码"
    )
    max_results: int = Field(
        default=10,
        description="最大搜索结果数"
    )
    safe_search: str = Field(
        default="moderate",
        description="安全搜索级别 (strict, moderate, off)"
    )
    auto_close: bool = Field(
        default=True,
        description="应用程序关闭时是否自动关闭 MCP 服务器"
    )

class LLMProviderSettings(BaseModel):
    """LLM 提供商设置"""
    # 每个提供商的默认模型
    openai_default_model: str = Field("gpt-4o", env="OPENAI_DEFAULT_MODEL")
    deepseek_default_model: str = Field("deepseek-chat", env="DEEPSEEK_DEFAULT_MODEL")
    qwen_default_model: str = Field("qwen-max", env="QWEN_DEFAULT_MODEL")
    google_default_model: str = Field("gemini-pro", env="GOOGLE_DEFAULT_MODEL")
    anthropic_default_model: str = Field("claude-3-sonnet-20240229", env="ANTHROPIC_DEFAULT_MODEL")
    xunfei_default_model: str = Field("general", env="XUNFEI_DEFAULT_MODEL")
    xai_default_model: str = Field("gemma-7b", env="XAI_DEFAULT_MODEL")

    # API密钥和端点
    # OpenAI
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    openai_api_base: Optional[str] = Field("https://api.openai.com/v1", env="OPENAI_API_BASE")
    openai_organization_id: Optional[str] = Field(None, env="OPENAI_ORGANIZATION_ID")

    # Deepseek
    deepseek_api_key: Optional[str] = Field(None, env="DEEPSEEK_API_KEY")
    deepseek_api_base: Optional[str] = Field("https://api.deepseek.com/v1", env="DEEPSEEK_API_BASE")

    # 阿里云千问
    qwen_api_key: Optional[str] = Field(None, env="QWEN_API_KEY")
    qwen_api_endpoint: Optional[str] = Field("https://dashscope.aliyuncs.com/api/v1", env="QWEN_API_ENDPOINT")

    # Google AI (Gemini)
    google_api_key: Optional[str] = Field(None, env="GOOGLE_API_KEY")
    google_project: Optional[str] = Field(None, env="GOOGLE_PROJECT")
    google_location: Optional[str] = Field("us-central1", env="GOOGLE_LOCATION")

    # Anthropic
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    anthropic_api_url: Optional[str] = Field("https://api.anthropic.com", env="ANTHROPIC_API_URL")

    # 讯飞
    xunfei_app_id: Optional[str] = Field(None, env="XUNFEI_APP_ID")
    xunfei_api_key: Optional[str] = Field(None, env="XUNFEI_API_KEY")
    xunfei_api_secret: Optional[str] = Field(None, env="XUNFEI_API_SECRET")

    # XAI
    xai_api_key: Optional[str] = Field(None, env="XAI_API_KEY")
    xai_api_base_url: Optional[str] = Field(None, env="XAI_API_BASE_URL")

    # HuggingFace
    hf_api_key: Optional[str] = Field(None, env="HF_API_KEY")

    # Azure OpenAI
    azure_openai_api_key: Optional[str] = Field(None, env="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: Optional[str] = Field(None, env="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_version: str = "2023-07-01-preview"

class VectorStoreSettings(BaseModel):
    """向量存储设置"""
    vector_store_dir: str = Field("./data/vector_store", env="VECTOR_STORE_DIR")
    vector_store_type: str = Field("faiss", env="VECTOR_STORE_TYPE")
    vector_search_top_k: int = 4

class DocumentProcessingSettings(BaseModel):
    """文档处理设置"""
    use_parallel_processing: bool = Field(True, env="DOC_USE_PARALLEL_PROCESSING")
    max_workers: int = Field(5, env="DOC_MAX_WORKERS")
    chunk_size_for_parallel: int = Field(10000, env="DOC_CHUNK_SIZE_FOR_PARALLEL")
    min_documents_for_parallel: int = Field(3, env="DOC_MIN_DOCUMENTS_FOR_PARALLEL")
    directory_processing_concurrency: int = Field(10, env="DOC_DIRECTORY_CONCURRENCY")

class CelerySettings(BaseModel):
    """Celery 设置"""
    broker_url: Optional[str] = Field(None, env="CELERY_BROKER_URL")
    result_backend: Optional[str] = Field(None, env="CELERY_RESULT_BACKEND")
    task_serializer: str = "json"
    result_serializer: str = "json"
    accept_content: List[str] = ["json"]
    timezone: str = "Asia/Shanghai"
    enable_utc: bool = True
    worker_concurrency: int = 4
    worker_prefetch_multiplier: int = 1

class MongoDBSettings(BaseModel):
    """MongoDB 设置"""
    uri: str = Field("mongodb://localhost:27017", env="MONGODB_URI")
    database: str = Field("dify_connect", env="MONGODB_DATABASE")
    conversations_collection: str = "conversations"
    messages_collection: str = "messages"
    max_pool_size: int = 100
    min_pool_size: int = 10
    connect_timeout_ms: int = 30000
    socket_timeout_ms: int = 30000

class Settings(BaseSettings):
    """应用程序设置"""

    # 应用程序基础设置
    APP_NAME: str = "Dify-Connect"
    APP_DESCRIPTION: str = "Dify-Connect API服务"
    APP_VERSION: str = "0.1.0"
    APP_ENV: str = Field("development", env="APP_ENV")
    DEBUG: bool = True
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")
    PORT: int = Field(8000, env="PORT")
    HOST: str = Field("0.0.0.0", env="HOST")

    # API设置
    API_PREFIX: str = "/api/v1"
    API_PORT: int = 8000
    API_HOST: str = "0.0.0.0"

    # CORS设置
    CORS_ORIGINS: List[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]

    # 数据目录
    DATA_DIR: str = os.path.join(ROOT_DIR, "data")
    KNOWLEDGE_BASE_DIR: str = os.path.join(DATA_DIR, "knowledge_base")
    CACHE_DIR: str = os.path.join(DATA_DIR, "cache")

    # 默认LLM设置
    DEFAULT_EMBEDDING_PROVIDER: str = Field("openai", env="DEFAULT_EMBEDDING_PROVIDER")
    DEFAULT_EMBEDDING_MODEL: str = Field("text-embedding-3-small", env="DEFAULT_EMBEDDING_MODEL")
    DEFAULT_VECTOR_STORE: str = "chroma"
    DEFAULT_TEXT_SPLITTER: str = "recursive"
    DEFAULT_LLM_PROVIDER: str = Field("openai", env="DEFAULT_LLM_PROVIDER")
    DEFAULT_LLM_MODEL: str = Field("gpt-3.5-turbo", env="DEFAULT_CHAT_MODEL")
    DEFAULT_TEMPERATURE: float = Field(0.7, env="DEFAULT_TEMPERATURE")
    DEFAULT_MAX_TOKENS: int = Field(2048, env="DEFAULT_MAX_TOKENS")

    # 嵌入模型
    OPENAI_EMBEDDING_MODEL: str = Field("text-embedding-3-small", env="OPENAI_EMBEDDING_MODEL")
    HUGGINGFACE_EMBEDDING_MODEL: str = Field("sentence-transformers/all-MiniLM-L6-v2", env="HUGGINGFACE_EMBEDDING_MODEL")

    # 文本分割器设置
    TEXT_SPLITTER_CHUNK_SIZE: int = 1000
    TEXT_SPLITTER_CHUNK_OVERLAP: int = 200

    # Redis 设置
    REDIS_HOST: str = Field("localhost", env="REDIS_HOST")
    REDIS_PORT: int = Field(6379, env="REDIS_PORT")
    REDIS_PASSWORD: Optional[str] = Field(None, env="REDIS_PASSWORD")
    REDIS_DB: int = Field(0, env="REDIS_DB")
    REDIS_RESULT_DB: int = Field(1, env="REDIS_RESULT_DB")

    # 子配置模型
    llm: LLMProviderSettings = LLMProviderSettings()
    vector_store: VectorStoreSettings = VectorStoreSettings()
    document_processing: DocumentProcessingSettings = DocumentProcessingSettings()
    celery: CelerySettings = CelerySettings()
    playwright: PlaywrightSettings = PlaywrightSettings()
    brave_search: BraveSearchSettings = BraveSearchSettings()
    mongodb: MongoDBSettings = MongoDBSettings()

    class Config:
        env_file = ".env"
        case_sensitive = True

    def create_dirs(self):
        """创建必要的目录"""
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.KNOWLEDGE_BASE_DIR, exist_ok=True)
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(self.vector_store.vector_store_dir), exist_ok=True)

    def update_from_env(self):
        """从环境变量更新配置"""
        # Playwright 设置
        if os.environ.get("PLAYWRIGHT_MCP_ENABLED"):
            self.playwright.enabled = os.environ.get("PLAYWRIGHT_MCP_ENABLED").lower() == "true"
        if os.environ.get("PLAYWRIGHT_MCP_HEADLESS"):
            self.playwright.headless = os.environ.get("PLAYWRIGHT_MCP_HEADLESS").lower() == "true"
        if os.environ.get("PLAYWRIGHT_MCP_PORT"):
            self.playwright.port = int(os.environ.get("PLAYWRIGHT_MCP_PORT"))
        if os.environ.get("PLAYWRIGHT_MCP_BROWSER"):
            self.playwright.browser = os.environ.get("PLAYWRIGHT_MCP_BROWSER")
        if os.environ.get("PLAYWRIGHT_MCP_AUTO_CLOSE"):
            self.playwright.auto_close = os.environ.get("PLAYWRIGHT_MCP_AUTO_CLOSE").lower() == "true"
        
        # Brave Search 设置
        if os.environ.get("BRAVE_SEARCH_MCP_ENABLED"):
            self.brave_search.enabled = os.environ.get("BRAVE_SEARCH_MCP_ENABLED").lower() == "true"
        if os.environ.get("BRAVE_SEARCH_API_KEY"):
            self.brave_search.api_key = os.environ.get("BRAVE_SEARCH_API_KEY")
        if os.environ.get("BRAVE_SEARCH_COUNTRY"):
            self.brave_search.country = os.environ.get("BRAVE_SEARCH_COUNTRY")
        if os.environ.get("BRAVE_SEARCH_MAX_RESULTS"):
            self.brave_search.max_results = int(os.environ.get("BRAVE_SEARCH_MAX_RESULTS"))

# 创建全局设置实例
settings = Settings()

# 从环境变量更新配置
settings.update_from_env()

# 确保必要的目录存在
settings.create_dirs()

# 依赖注入函数
def get_settings():
    """获取设置实例的依赖注入函数"""
    return settings