#!/bin/bash
# Poetry 安装脚本 - Shell 版本

# 获取项目根目录
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "=== Dify-Connect Poetry 安装脚本 ==="

# 检查 Poetry 是否已安装
if ! command -v poetry &> /dev/null; then
    echo "未检测到 Poetry，将为您安装..."
    
    # 安装 Poetry
    curl -sSL https://install.python-poetry.org | python3 -
    
    # 添加 Poetry 到 PATH
    export PATH="$HOME/.poetry/bin:$PATH"
    
    # 再次检查 Poetry 是否可用
    if ! command -v poetry &> /dev/null; then
        echo "Poetry 安装后仍无法使用。请手动将 Poetry 添加到 PATH 环境变量，然后重新运行此脚本。"
        echo "Poetry 可能安装在: $HOME/.poetry/bin"
        exit 1
    fi
else
    echo "检测到 Poetry 已安装"
fi

# 检查 pyproject.toml 是否存在
if [ ! -f "pyproject.toml" ]; then
    echo "错误: 在项目根目录中找不到 pyproject.toml 文件"
    exit 1
fi

# 安装依赖
echo "正在安装项目依赖..."
poetry install

echo -e "\n=== 安装完成 ==="
echo "您现在可以使用以下命令激活虚拟环境并运行项目:"
echo "  poetry shell"
echo "  python scripts/start_all.py"
echo -e "\n或者直接运行:"
echo "  poetry run python scripts/start_all.py"
