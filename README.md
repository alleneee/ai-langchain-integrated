## Dify-Connect

一个统一的LLM连接服务，支持多种LLM提供商和服务集成。

### 项目结构

```
dify-connect/
├── src/                        # 源代码目录
│   ├── api/                    # API路由和接口定义
│   ├── celery_app/             # Celery 异步任务相关
│   │   ├── tasks/             # Celery 任务定义
│   │   ├── celery_app.py       # Celery 应用实例
│   │   └── celery_config.py    # Celery 配置
│   ├── config/                 # 配置管理
│   ├── factories/              # 工厂类
│   ├── schemas/                # 数据模型定义
│   ├── services/               # 服务实现层
│   └── utils/                  # 工具函数
├── scripts/                    # 脚本文件
│   ├── celery_worker.py        # Celery Worker 启动脚本
│   ├── celery_flower.py        # Celery Flower 监控脚本
│   └── migrate_to_pydantic_v2.py # Pydantic v1 到 v2 迁移脚本
├── docker/                     # Docker 相关文件
│   └── docker-compose-celery.yml # Celery Docker Compose 配置
├── docs/                       # 文档
│   ├── celery-integration.md   # Celery 集成文档
│   ├── document-formats.md     # 文档格式支持文档
│   └── langchain_0_3_migration.md # LangChain 0.3 迁移指南
├── examples/                   # 使用示例
├── tests/                      # 测试代码
├── requirements.txt            # 依赖包列表
├── .env.example                # 环境变量示例
├── .gitignore                  # Git忽略文件
└── README.md                   # 项目文档
```

### 文档

- [快速启动指南](docs/quick-start.md) - 如何使用一键启动脚本启动项目
- [Celery 集成文档](docs/celery-integration.md) - 如何使用 Celery 实现异步文档处理
- [文档格式支持文档](docs/document-formats.md) - 支持的文档格式和使用方法
- [LangChain 0.3 迁移指南](docs/langchain_0_3_migration.md) - 如何升级到 LangChain 0.3 版本

### 安装依赖

#### 使用 Poetry（推荐）

我们提供了一键安装脚本：

```bash
# Linux/macOS
bash scripts/setup_poetry.sh

# Windows
scripts\setup_poetry.bat

# 或者使用 Python 脚本（跨平台）
python scripts/setup_poetry.py
```

如果您已经安装了 Poetry，可以直接运行：

```bash
poetry install
```

#### 使用 pip

```bash
pip install -r requirements.txt
```

更多详细信息，请参考[快速启动指南](docs/quick-start.md)。

### LangChain 0.3 更新

本项目已升级至 LangChain 0.3。主要变更包括：

- 全面升级到 Pydantic 2，不再支持 Pydantic 1
- 更新了集成包的导入路径，如 langchain-openai、langchain-google-genai 等
- 不再支持 Python 3.8

如需迁移现有代码，请使用迁移脚本：

```bash
python scripts/migrate_to_pydantic_v2.py src/ --dry-run  # 预览更改
python scripts/migrate_to_pydantic_v2.py src/            # 应用更改
```

详细迁移指南请参考[LangChain 0.3 迁移指南](docs/langchain_0_3_migration.md)。

### 配置

在项目根目录创建`.env`文件，参考`.env.example`进行配置：

```
# 服务器配置
HOST=0.0.0.0
PORT=8000

# OpenAI配置
OPENAI_API_KEY=your_openai_api_key
OPENAI_API_BASE=https://api.openai.com/v1

# 其他LLM提供商配置
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_API_BASE=https://api.deepseek.com/v1

QWEN_API_KEY=your_qwen_api_key
QWEN_API_BASE=https://dashscope.aliyuncs.com/api/v1
```

### 运行服务

```bash
# 使用Poetry
poetry run python -m app.main

# 或直接使用Python
python -m app.main
```

服务将在<http://localhost:8000启动，可通过http://localhost:8000/docs访问API文档。>

### API示例

#### 聊天API

```python
import requests
import json

url = "http://localhost:8000/api/v1/chat/messages"
headers = {"Content-Type": "application/json"}
data = {
    "user": "user123",
    "inputs": {},
    "query": "你好，请介绍一下自己",
    "conversation_id": None,
    "response_mode": "blocking"
}

response = requests.post(url, headers=headers, data=json.dumps(data))
print(response.json())
```

### 贡献指南

欢迎提交Pull Request或Issue来完善本项目。贡献前请先阅读以下规范：

1. 使用统一的代码风格，遵循PEP 8规范
2. 提交前进行代码格式化
3. 新功能需编写测试用例
4. 保持良好的文档和注释

### 许可证

MIT
