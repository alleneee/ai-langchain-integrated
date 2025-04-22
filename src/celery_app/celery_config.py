"""
Celery 配置模块

定义 Celery 的配置参数
"""

import os
import sys
from kombu import Queue, Exchange

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 导入设置
from src.config.settings import settings

# 消息代理 URL
if settings.CELERY_BROKER_URL:
    broker_url = settings.CELERY_BROKER_URL
else:
    redis_password = f":{settings.REDIS_PASSWORD}@" if settings.REDIS_PASSWORD else ""
    broker_url = f"redis://{redis_password}{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}"

# 结果后端 URL
if settings.CELERY_RESULT_BACKEND:
    result_backend = settings.CELERY_RESULT_BACKEND
else:
    redis_password = f":{settings.REDIS_PASSWORD}@" if settings.REDIS_PASSWORD else ""
    result_backend = f"redis://{redis_password}{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_RESULT_DB}"

# 任务序列化格式
task_serializer = 'json'

# 结果序列化格式
result_serializer = 'json'

# 接受的内容类型
accept_content = ['json']

# 时区
timezone = 'Asia/Shanghai'

# 启用 UTC
enable_utc = True

# 任务结果过期时间（秒）
result_expires = 60 * 60 * 24  # 24小时

# 工作进程并发数
worker_concurrency = os.cpu_count()

# 每个工作进程执行的最大任务数
worker_max_tasks_per_child = 1000

# 任务软时间限制（秒）
task_soft_time_limit = 60 * 30  # 30分钟

# 任务硬时间限制（秒）
task_time_limit = 60 * 60  # 1小时

# 任务默认队列
task_default_queue = 'default'

# 任务默认交换机
task_default_exchange = 'default'

# 任务默认路由键
task_default_routing_key = 'default'

# 定义队列
task_queues = (
    Queue('default', Exchange('default'), routing_key='default'),
    Queue('documents', Exchange('documents'), routing_key='documents'),
    Queue('high_priority', Exchange('high_priority'), routing_key='high_priority'),
)

# 任务路由
task_routes = {
    'src.celery_app.tasks.document_tasks.*': {'queue': 'documents'},
}

# 任务注解
task_annotations = {
    'src.celery_app.tasks.document_tasks.process_document': {'rate_limit': '10/m'},
}

# 任务发送重试次数
task_publish_retry_policy = {
    'max_retries': 3,
    'interval_start': 0,
    'interval_step': 0.2,
    'interval_max': 0.5,
}

# 启用任务事件
worker_send_task_events = True

# 启用心跳
worker_enable_remote_control = True
