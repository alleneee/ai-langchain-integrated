# Celery 集成：异步文档处理

本文档介绍了如何使用 Celery 实现异步文档处理功能。

## 1. 概述

对于处理大型文档或大量文档的场景，同步处理可能会导致 API 请求超时。通过集成 Celery，我们可以将文档处理任务放入后台异步执行，提高系统的响应性和可靠性。

**系统现已支持的异步处理功能：**
- 异步文档处理：处理、分割并存储普通文档
- 异步向量知识库处理：将文档添加到向量数据库（例如 ChromaDB）以便于语义搜索

## 2. 架构

异步文档处理的架构如下：

```
客户端 -> API 服务 -> Celery 任务队列 -> Celery Worker -> 文档处理服务/向量知识库服务
                                      -> Redis 结果后端 -> 客户端查询结果
```

主要组件：
- **API 服务**：接收客户端请求，提交任务到 Celery 队列
- **Celery 任务队列**：存储待处理的任务
- **Celery Worker**：执行文档处理任务
- **Redis**：作为消息代理和结果后端
- **文档处理服务**：实际处理文档的服务
- **向量知识库服务**：处理文档并存储到向量数据库（例如 ChromaDB）

## 3. 安装依赖

Celery 相关依赖已经包含在主 requirements.txt 文件中，只需要安装主依赖即可：

```bash
pip install -r requirements.txt
```

## 4. 启动服务

### 4.1 使用 Docker Compose

```bash
docker-compose -f docker/docker-compose-celery.yml up
```

### 4.2 手动启动

1. 启动 Redis 服务：

```bash
redis-server
```

2. 启动 Celery Worker：

```bash
python scripts/celery_worker.py
```

3. 启动 Flower 监控（可选）：

```bash
python scripts/celery_flower.py
```

4. 启动 API 服务：

```bash
uvicorn src.main:app --reload
```

## 5. API 使用

### 5.1 文档处理 API

#### 5.1.1 异步上传并处理文档

```
POST /documents/async/upload
```

参数：
- `file`：要上传的文件
- `split`：是否分割文档（可选，默认为 false）
- `chunk_size`：分割大小（可选，默认为 1000）
- `chunk_overlap`：分割重叠大小（可选，默认为 200）

响应：
```json
{
  "task_id": "任务ID",
  "status": "PENDING",
  "message": "文档处理任务已提交"
}
```

#### 5.1.2 异步处理 URL 或文件路径

```
POST /documents/async/process
```

参数：
- `source`：文档源（URL 或文件路径）
- `split`：是否分割文档（可选，默认为 false）
- `chunk_size`：分割大小（可选，默认为 1000）
- `chunk_overlap`：分割重叠大小（可选，默认为 200）

响应：
```json
{
  "task_id": "任务ID",
  "status": "PENDING",
  "message": "文档处理任务已提交"
}
```

#### 5.1.3 获取任务状态

```
GET /documents/async/tasks/{task_id}
```

响应：
```json
{
  "task_id": "任务ID",
  "status": "SUCCESS",
  "progress": 100,
  "result": {
    "status": "success",
    "filename": "文件名",
    "document_count": 5,
    "documents": [...]
  }
}
```

可能的状态：
- `PENDING`：任务等待中
- `STARTED`：任务进行中
- `SUCCESS`：任务成功完成
- `FAILURE`：任务失败

### 5.2 向量知识库 API

#### 5.2.1 异步添加文档到知识库

```
POST /vector-kb/async/documents
```

参数：
```json
{
  "source_path": "/path/to/documents",
  "collection_name": "my_collection",
  "chunk_size": 1000,
  "chunk_overlap": 200
}
```

响应：
```json
{
  "task_id": "任务ID",
  "status": "PENDING",
  "message": "文档处理任务已提交"
}
```

#### 5.2.2 批量异步添加多个文档到知识库

```
POST /vector-kb/async/batch
```

参数：
```json
{
  "source_paths": ["/path/to/doc1", "/path/to/doc2", "/path/to/doc3"],
  "collection_name": "my_collection",
  "chunk_size": 1000,
  "chunk_overlap": 200
}
```

响应：
```json
{
  "task_id": "任务ID",
  "status": "PENDING",
  "message": "批量文档处理任务已提交，将处理 3 个源"
}
```

#### 5.2.3 异步上传并处理文档到知识库

```
POST /vector-kb/async/upload
```

参数：
- `file`：要上传的文件
- `collection_name`：集合名称（可选）
- `chunk_size`：分割大小（可选，默认为 1000）
- `chunk_overlap`：分割重叠大小（可选，默认为 200）

响应：
```json
{
  "task_id": "任务ID",
  "status": "PENDING",
  "message": "文件 example.pdf 已上传并提交处理"
}
```

#### 5.2.4 获取任务状态

```
GET /vector-kb/async/tasks/{task_id}
```

响应：
```json
{
  "task_id": "任务ID",
  "status": "SUCCESS",
  "progress": 100,
  "result": {
    "status": "success",
    "message": "文档已成功添加到知识库",
    "source_path": "/path/to/documents",
    "collection_name": "my_collection",
    "process_time": 5.67
  }
}
```

## 6. 示例代码

### 6.1 文档处理示例

参见 `examples/async_document_processing_example.py`：

```python
import requests
import time

# 提交异步处理任务
data = {'source': 'https://example.com', 'split': True}
response = requests.post('http://localhost:8000/documents/async/process', data=data)
task_data = response.json()
task_id = task_data["task_id"]

# 轮询任务状态
while True:
    status_response = requests.get(f'http://localhost:8000/documents/async/tasks/{task_id}')
    status_data = status_response.json()
    
    if status_data['status'] == 'SUCCESS':
        result = status_data['result']
        break
    elif status_data['status'] == 'FAILURE':
        print(f"任务失败: {status_data.get('error')}")
        break
    
    time.sleep(1)
```

### 6.2 向量知识库处理示例

```python
import requests
import time
import json

# 提交异步处理任务
data = {
    "source_path": "./documents",
    "collection_name": "my_documents",
    "chunk_size": 1000,
    "chunk_overlap": 200
}
response = requests.post('http://localhost:8000/vector-kb/async/documents', json=data)
task_data = response.json()
task_id = task_data["task_id"]
print(f"任务已提交，ID: {task_id}")

# 轮询任务状态
while True:
    status_response = requests.get(f'http://localhost:8000/vector-kb/async/tasks/{task_id}')
    status_data = status_response.json()
    
    if 'progress' in status_data and status_data['progress'] is not None:
        print(f"进度: {status_data['progress']}%")
    
    if status_data['status'] == 'SUCCESS':
        print("任务成功完成!")
        result = status_data['result']
        print(f"处理结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
        break
    elif status_data['status'] == 'FAILURE':
        print(f"任务失败: {status_data.get('error')}")
        break
    
    time.sleep(2)
```

## 7. 监控

Celery Flower 提供了一个 Web 界面来监控 Celery 任务：

```
http://localhost:5555
```

通过 Flower，您可以：
- 查看任务队列状态
- 监控任务执行情况
- 查看任务历史记录
- 取消或重试任务

## 8. 配置

Celery 配置位于 `src/celery_app/celery_config.py`，您可以根据需要调整以下参数：

- `broker_url`：消息代理 URL
- `result_backend`：结果后端 URL
- `worker_concurrency`：工作进程并发数
- `task_soft_time_limit`：任务软时间限制
- `task_time_limit`：任务硬时间限制

任务路由可以调整为：

```python
# 任务路由
task_routes = {
    'src.celery_app.tasks.document_tasks.*': {'queue': 'documents'},
    'src.celery_app.tasks.kb_tasks.*': {'queue': 'knowledge_base'},
}
```

## 9. 故障排除

1. **任务卡在 PENDING 状态**：
   - 检查 Celery Worker 是否正在运行
   - 检查 Redis 服务是否可用

2. **任务失败**：
   - 查看 Celery Worker 日志
   - 检查任务状态响应中的错误信息

3. **性能问题**：
   - 增加 Worker 数量
   - 调整 `worker_concurrency` 参数
   - 为不同类型的任务使用不同的队列

## 10. 向量知识库特定问题

1. **文档处理速度慢**：
   - 对于大型文档集合，可以增加 Worker 数量
   - 使用 `batch_add_documents_to_kb` 任务分批处理文档
   - 将 ChromaDB 的持久化目录设置在 SSD 上以提高性能

2. **向量存储错误**：
   - 确保 OpenAI API 密钥已正确设置在环境变量中
   - 检查向量存储目录的权限
   - 查看 Worker 日志中的详细错误信息
