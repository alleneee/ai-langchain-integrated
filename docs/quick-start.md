# 快速启动指南

本文档介绍了如何使用一键启动脚本快速启动整个项目，包括 Redis、Celery Worker、Flower 和 FastAPI 服务。

## 1. 环境准备

### 1.1 安装依赖

有两种方式安装依赖：

#### 使用 Poetry（推荐）

Poetry 是一个现代化的 Python 依赖管理和打包工具，可以更好地管理项目依赖和虚拟环境。

我们提供了一键安装脚本：

```bash
# Linux/macOS
bash scripts/setup_poetry.sh

# Windows
scripts\setup_poetry.bat

# 或者使用 Python 脚本（跨平台）
python scripts/setup_poetry.py
```

安装完成后，您可以使用以下命令激活虚拟环境：

```bash
poetry shell
```

#### 使用 pip

如果您更喜欢使用传统的 pip，可以运行：

```bash
pip install -r requirements.txt
```

### 1.2 配置环境变量

复制 `.env.example` 文件并重命名为 `.env`，然后根据需要修改其中的配置：

```bash
cp .env.example .env
```

特别注意以下 Redis 和 Celery 相关的配置：

```
# Redis 配置
REDIS_HOST=localhost
REDIS_PORT=6379
# REDIS_PASSWORD=your-redis-password  # 如果需要密码，请取消注释
REDIS_DB=0
REDIS_RESULT_DB=1

# Celery 配置
# 如果需要自定义 Celery 配置，请取消下面的注释
# CELERY_BROKER_URL=redis://localhost:6379/0
# CELERY_RESULT_BACKEND=redis://localhost:6379/1
```

## 2. 使用一键启动脚本

### 2.1 Python 版本（跨平台）

```bash
python scripts/start_all.py
```

可用选项：
- `--no-redis`: 不启动 Redis 服务
- `--no-worker`: 不启动 Celery Worker
- `--no-flower`: 不启动 Celery Flower
- `--no-api`: 不启动 API 服务
- `--no-reload`: API 服务不使用热重载

例如，如果您已经有一个运行中的 Redis 服务，可以使用：

```bash
python scripts/start_all.py --no-redis
```

### 2.2 Shell 脚本版本（Unix/Linux/macOS）

```bash
bash scripts/start_all.sh
```

可用选项与 Python 版本相同。

### 2.3 批处理脚本版本（Windows）

```bash
scripts\start_all.bat
```

可用选项与 Python 版本相同。

## 3. 单独启动各个服务

如果您想单独启动各个服务，可以使用以下命令：

### 3.1 启动 Redis

```bash
redis-server --port 6379
```

### 3.2 启动 Celery Worker

```bash
python scripts/celery_worker.py
```

### 3.3 启动 Celery Flower 监控

```bash
python scripts/celery_flower.py
```

### 3.4 启动 API 服务

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

## 4. 使用 Docker Compose

您也可以使用 Docker Compose 一键启动所有服务：

```bash
docker-compose -f docker/docker-compose-celery.yml up
```

如果只想启动特定服务，可以使用：

```bash
docker-compose -f docker/docker-compose-celery.yml up redis celery_worker
```

## 5. 访问服务

启动所有服务后，您可以通过以下地址访问各个服务：

- API 服务：http://localhost:8000
- API 文档：http://localhost:8000/docs
- Celery Flower 监控：http://localhost:5555

## 6. 故障排除

### 6.1 Redis 连接问题

如果遇到 Redis 连接问题，请检查：
- Redis 服务是否正在运行
- Redis 主机和端口配置是否正确
- 如果设置了密码，密码是否正确

### 6.2 Celery 任务不执行

如果 Celery 任务不执行，请检查：
- Celery Worker 是否正在运行
- Redis 连接是否正常
- 任务是否正确提交

### 6.3 API 服务启动失败

如果 API 服务启动失败，请检查：
- 端口是否被占用
- 环境变量是否正确配置
- 依赖是否完整安装
