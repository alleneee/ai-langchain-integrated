@echo off
:: 一键启动脚本 - Windows 批处理版本

:: 获取项目根目录
set ROOT_DIR=%~dp0..
cd %ROOT_DIR%

:: 加载环境变量
if exist .env (
    for /f "tokens=*" %%a in (.env) do (
        set %%a
    )
)

:: 设置默认值
if not defined REDIS_HOST set REDIS_HOST=localhost
if not defined REDIS_PORT set REDIS_PORT=6379
if not defined REDIS_DB set REDIS_DB=0
if not defined REDIS_RESULT_DB set REDIS_RESULT_DB=1

:: 解析命令行参数
set NO_REDIS=false
set NO_WORKER=false
set NO_FLOWER=false
set NO_API=false
set NO_RELOAD=false

:parse_args
if "%~1"=="" goto :end_parse_args
if "%~1"=="--no-redis" set NO_REDIS=true
if "%~1"=="--no-worker" set NO_WORKER=true
if "%~1"=="--no-flower" set NO_FLOWER=true
if "%~1"=="--no-api" set NO_API=true
if "%~1"=="--no-reload" set NO_RELOAD=true
shift
goto :parse_args
:end_parse_args

:: 启动 Redis
if %NO_REDIS%==false (
    echo 正在启动 Redis 服务...
    
    :: 检查 Redis 是否已经在运行
    redis-cli -h %REDIS_HOST% -p %REDIS_PORT% ping >nul 2>&1
    if %errorlevel%==0 (
        echo Redis 服务已经在运行
    ) else (
        :: 启动 Redis 服务
        where redis-server >nul 2>&1
        if %errorlevel%==0 (
            start "Redis Server" redis-server --port %REDIS_PORT%
            echo Redis 服务已启动
            :: 等待 Redis 启动
            timeout /t 2 >nul
        ) else (
            echo 启动 Redis 服务失败
            echo 请确保已安装 Redis 或手动启动 Redis 服务
        )
    )
)

:: 设置 Celery 环境变量
set CELERY_BROKER_URL=redis://%REDIS_HOST%:%REDIS_PORT%/%REDIS_DB%
set CELERY_RESULT_BACKEND=redis://%REDIS_HOST%:%REDIS_PORT%/%REDIS_RESULT_DB%

:: 启动 Celery Worker
if %NO_WORKER%==false (
    echo 正在启动 Celery Worker...
    start "Celery Worker" python %ROOT_DIR%\scripts\celery_worker.py
    echo Celery Worker 已启动
)

:: 启动 Celery Flower
if %NO_FLOWER%==false (
    echo 正在启动 Celery Flower 监控...
    start "Celery Flower" python %ROOT_DIR%\scripts\celery_flower.py
    echo Celery Flower 已启动
)

:: 启动 API 服务
if %NO_API%==false (
    echo 正在启动 API 服务...
    
    if %NO_RELOAD%==true (
        start "FastAPI Server" uvicorn src.main:app --host 0.0.0.0 --port 8000
    ) else (
        start "FastAPI Server" uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
    )
    
    echo API 服务已启动
)

echo.
echo 所有服务已启动，请在各自的窗口中关闭服务
echo 或者按 Ctrl+C 然后输入 Y 来关闭当前窗口

:: 保持窗口打开
pause
