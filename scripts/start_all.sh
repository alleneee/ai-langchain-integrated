#!/bin/bash
# 一键启动脚本 - Shell 版本

# 获取项目根目录
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# 加载环境变量
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# 进程 ID 列表
PIDS=()

# 清理函数
cleanup() {
    echo -e "\n正在关闭所有服务..."
    
    # 终止所有进程
    for pid in "${PIDS[@]}"; do
        if ps -p $pid > /dev/null; then
            kill $pid
        fi
    done
    
    echo "所有服务已关闭"
    exit 0
}

# 捕获 Ctrl+C
trap cleanup INT TERM

# 启动 Redis
start_redis() {
    echo "正在启动 Redis 服务..."
    
    # 检查 Redis 是否已经在运行
    if command -v redis-cli > /dev/null && redis-cli -h ${REDIS_HOST:-localhost} -p ${REDIS_PORT:-6379} ping > /dev/null 2>&1; then
        echo "Redis 服务已经在运行"
    else
        # 启动 Redis 服务
        if command -v redis-server > /dev/null; then
            redis-server --port ${REDIS_PORT:-6379} &
            REDIS_PID=$!
            PIDS+=($REDIS_PID)
            echo "Redis 服务已启动，PID: $REDIS_PID"
            # 等待 Redis 启动
            sleep 2
        else
            echo "启动 Redis 服务失败"
            echo "请确保已安装 Redis 或手动启动 Redis 服务"
        fi
    fi
}

# 启动 Celery Worker
start_celery_worker() {
    echo "正在启动 Celery Worker..."
    export CELERY_BROKER_URL="redis://${REDIS_HOST:-localhost}:${REDIS_PORT:-6379}/${REDIS_DB:-0}"
    export CELERY_RESULT_BACKEND="redis://${REDIS_HOST:-localhost}:${REDIS_PORT:-6379}/${REDIS_RESULT_DB:-1}"
    
    python "$ROOT_DIR/scripts/celery_worker.py" &
    WORKER_PID=$!
    PIDS+=($WORKER_PID)
    echo "Celery Worker 已启动，PID: $WORKER_PID"
}

# 启动 Celery Flower
start_celery_flower() {
    echo "正在启动 Celery Flower 监控..."
    export CELERY_BROKER_URL="redis://${REDIS_HOST:-localhost}:${REDIS_PORT:-6379}/${REDIS_DB:-0}"
    export CELERY_RESULT_BACKEND="redis://${REDIS_HOST:-localhost}:${REDIS_PORT:-6379}/${REDIS_RESULT_DB:-1}"
    
    python "$ROOT_DIR/scripts/celery_flower.py" &
    FLOWER_PID=$!
    PIDS+=($FLOWER_PID)
    echo "Celery Flower 已启动，PID: $FLOWER_PID"
}

# 启动 API 服务
start_api_server() {
    echo "正在启动 API 服务..."
    
    if [ "$1" = "--no-reload" ]; then
        uvicorn src.main:app --host 0.0.0.0 --port 8000 &
    else
        uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload &
    fi
    
    API_PID=$!
    PIDS+=($API_PID)
    echo "API 服务已启动，PID: $API_PID"
}

# 解析命令行参数
NO_REDIS=false
NO_WORKER=false
NO_FLOWER=false
NO_API=false
NO_RELOAD=false

for arg in "$@"; do
    case $arg in
        --no-redis)
            NO_REDIS=true
            ;;
        --no-worker)
            NO_WORKER=true
            ;;
        --no-flower)
            NO_FLOWER=true
            ;;
        --no-api)
            NO_API=true
            ;;
        --no-reload)
            NO_RELOAD=true
            ;;
    esac
done

# 启动服务
if [ "$NO_REDIS" = false ]; then
    start_redis
fi

if [ "$NO_WORKER" = false ]; then
    start_celery_worker
fi

if [ "$NO_FLOWER" = false ]; then
    start_celery_flower
fi

if [ "$NO_API" = false ]; then
    if [ "$NO_RELOAD" = true ]; then
        start_api_server --no-reload
    else
        start_api_server
    fi
fi

echo -e "\n所有服务已启动，按 Ctrl+C 停止所有服务"

# 等待任意子进程结束
wait

# 清理
cleanup
