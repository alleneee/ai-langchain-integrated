@echo off
:: Poetry 安装脚本 - Windows 批处理版本

:: 获取项目根目录
set ROOT_DIR=%~dp0..
cd %ROOT_DIR%

echo === Dify-Connect Poetry 安装脚本 ===

:: 检查 Poetry 是否已安装
where poetry >nul 2>&1
if %errorlevel% neq 0 (
    echo 未检测到 Poetry，将为您安装...
    
    :: 安装 Poetry
    powershell -Command "(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -"
    
    :: 添加 Poetry 到 PATH
    set PATH=%USERPROFILE%\.poetry\bin;%PATH%
    
    :: 再次检查 Poetry 是否可用
    where poetry >nul 2>&1
    if %errorlevel% neq 0 (
        echo Poetry 安装后仍无法使用。请手动将 Poetry 添加到 PATH 环境变量，然后重新运行此脚本。
        echo Poetry 可能安装在: %USERPROFILE%\.poetry\bin
        exit /b 1
    )
) else (
    echo 检测到 Poetry 已安装
)

:: 检查 pyproject.toml 是否存在
if not exist "pyproject.toml" (
    echo 错误: 在项目根目录中找不到 pyproject.toml 文件
    exit /b 1
)

:: 安装依赖
echo 正在安装项目依赖...
poetry install

echo.
echo === 安装完成 ===
echo 您现在可以使用以下命令激活虚拟环境并运行项目:
echo   poetry shell
echo   python scripts/start_all.py
echo.
echo 或者直接运行:
echo   poetry run python scripts/start_all.py
