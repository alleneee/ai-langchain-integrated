#!/usr/bin/env python
"""
一键启动脚本

启动整个项目，包括 Redis、Celery Worker、Flower 和 FastAPI 服务
"""

import os
import sys
import time
import signal
import subprocess
import argparse
from pathlib import Path

# 添加项目根目录到 Python 路径
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

# 导入配置
from src.config.settings import settings

# 进程列表
processes = []

def start_redis():
    """启动 Redis 服务"""
    print("正在启动 Redis 服务...")

    # 检查 Redis 是否已经在运行
    try:
        import redis
        r = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            socket_connect_timeout=1
        )
        r.ping()
        print("Redis 服务已经在运行")
        return None
    except:
        pass

    # 启动 Redis 服务
    try:
        redis_process = subprocess.Popen(
            ["redis-server", "--port", str(settings.REDIS_PORT)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        print(f"Redis 服务已启动，PID: {redis_process.pid}")
        return redis_process
    except Exception as e:
        print(f"启动 Redis 服务失败: {str(e)}")
        print("请确保已安装 Redis 或手动启动 Redis 服务")
        return None

def start_celery_worker():
    """启动 Celery Worker"""
    print("正在启动 Celery Worker...")

    # 检查是否在 Poetry 环境中
    in_poetry = os.environ.get("POETRY_ACTIVE") == "1"

    if in_poetry:
        # 在 Poetry 环境中直接运行 Python 脚本
        cmd = [sys.executable, os.path.join(ROOT_DIR, "scripts/celery_worker.py")]
    else:
        # 尝试使用 Poetry 运行
        try:
            subprocess.run(["poetry", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            cmd = ["poetry", "run", "python", os.path.join(ROOT_DIR, "scripts/celery_worker.py")]
        except (subprocess.CalledProcessError, FileNotFoundError):
            # 如果 Poetry 不可用，直接使用 Python
            cmd = [sys.executable, os.path.join(ROOT_DIR, "scripts/celery_worker.py")]

    worker_process = subprocess.Popen(
        cmd,
        env=dict(os.environ,
                 CELERY_BROKER_URL=f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}",
                 CELERY_RESULT_BACKEND=f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_RESULT_DB}")
    )
    print(f"Celery Worker 已启动，PID: {worker_process.pid}")
    return worker_process

def start_celery_flower():
    """启动 Celery Flower 监控"""
    print("正在启动 Celery Flower 监控...")

    # 检查是否在 Poetry 环境中
    in_poetry = os.environ.get("POETRY_ACTIVE") == "1"

    if in_poetry:
        # 在 Poetry 环境中直接运行 Python 脚本
        cmd = [sys.executable, os.path.join(ROOT_DIR, "scripts/celery_flower.py")]
    else:
        # 尝试使用 Poetry 运行
        try:
            subprocess.run(["poetry", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            cmd = ["poetry", "run", "python", os.path.join(ROOT_DIR, "scripts/celery_flower.py")]
        except (subprocess.CalledProcessError, FileNotFoundError):
            # 如果 Poetry 不可用，直接使用 Python
            cmd = [sys.executable, os.path.join(ROOT_DIR, "scripts/celery_flower.py")]

    flower_process = subprocess.Popen(
        cmd,
        env=dict(os.environ,
                 CELERY_BROKER_URL=f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}",
                 CELERY_RESULT_BACKEND=f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_RESULT_DB}")
    )
    print(f"Celery Flower 已启动，PID: {flower_process.pid}")
    return flower_process

def start_api_server(reload=True):
    """启动 FastAPI 服务"""
    print("正在启动 API 服务...")

    # 检查是否在 Poetry 环境中
    in_poetry = os.environ.get("POETRY_ACTIVE") == "1"

    base_cmd = ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
    if reload:
        base_cmd.append("--reload")

    if in_poetry:
        # 在 Poetry 环境中直接运行命令
        cmd = base_cmd
    else:
        # 尝试使用 Poetry 运行
        try:
            subprocess.run(["poetry", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            cmd = ["poetry", "run"] + base_cmd
        except (subprocess.CalledProcessError, FileNotFoundError):
            # 如果 Poetry 不可用，直接运行命令
            cmd = base_cmd

    api_process = subprocess.Popen(cmd)
    print(f"API 服务已启动，PID: {api_process.pid}")
    return api_process

def cleanup(sig=None, frame=None):
    """清理所有进程"""
    print("\n正在关闭所有服务...")

    for process in processes:
        if process and process.poll() is None:  # 如果进程还在运行
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

    print("所有服务已关闭")
    sys.exit(0)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="一键启动项目")
    parser.add_argument("--no-redis", action="store_true", help="不启动 Redis 服务")
    parser.add_argument("--no-worker", action="store_true", help="不启动 Celery Worker")
    parser.add_argument("--no-flower", action="store_true", help="不启动 Celery Flower")
    parser.add_argument("--no-api", action="store_true", help="不启动 API 服务")
    parser.add_argument("--no-reload", action="store_true", help="API 服务不使用热重载")
    args = parser.parse_args()

    # 注册信号处理
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        # 启动 Redis
        if not args.no_redis:
            redis_process = start_redis()
            if redis_process:
                processes.append(redis_process)
                # 等待 Redis 启动
                time.sleep(2)

        # 启动 Celery Worker
        if not args.no_worker:
            worker_process = start_celery_worker()
            processes.append(worker_process)

        # 启动 Celery Flower
        if not args.no_flower:
            flower_process = start_celery_flower()
            processes.append(flower_process)

        # 启动 API 服务
        if not args.no_api:
            api_process = start_api_server(not args.no_reload)
            processes.append(api_process)

        print("\n所有服务已启动，按 Ctrl+C 停止所有服务")

        # 等待所有进程结束
        for process in processes:
            if process:
                process.wait()

    except KeyboardInterrupt:
        cleanup()
    except Exception as e:
        print(f"发生错误: {str(e)}")
        cleanup()

if __name__ == "__main__":
    main()
