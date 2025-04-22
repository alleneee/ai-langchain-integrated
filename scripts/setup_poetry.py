#!/usr/bin/env python
"""
Poetry 安装脚本

帮助用户使用 Poetry 安装项目依赖
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_poetry_installed():
    """检查 Poetry 是否已安装"""
    try:
        subprocess.run(["poetry", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_poetry():
    """安装 Poetry"""
    print("正在安装 Poetry...")
    
    if platform.system() == "Windows":
        # Windows 安装方法
        command = "(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -"
        subprocess.run(["powershell", "-Command", command], check=True)
    else:
        # Unix/Linux/macOS 安装方法
        command = "curl -sSL https://install.python-poetry.org | python3 -"
        subprocess.run(command, shell=True, check=True)
    
    print("Poetry 安装完成！")

def setup_project():
    """设置项目"""
    # 获取项目根目录
    root_dir = Path(__file__).parent.parent.absolute()
    os.chdir(root_dir)
    
    print(f"项目根目录: {root_dir}")
    
    # 检查 pyproject.toml 是否存在
    if not os.path.exists("pyproject.toml"):
        print("错误: 在项目根目录中找不到 pyproject.toml 文件")
        sys.exit(1)
    
    # 安装依赖
    print("正在安装项目依赖...")
    subprocess.run(["poetry", "install"], check=True)
    
    print("依赖安装完成！")

def main():
    """主函数"""
    print("=== Dify-Connect Poetry 安装脚本 ===")
    
    # 检查 Poetry 是否已安装
    if not check_poetry_installed():
        print("未检测到 Poetry，将为您安装...")
        install_poetry()
        
        # 添加 Poetry 到 PATH
        if platform.system() == "Windows":
            poetry_path = os.path.expanduser("~\\.poetry\\bin")
        else:
            poetry_path = os.path.expanduser("~/.poetry/bin")
        
        if poetry_path not in os.environ["PATH"]:
            os.environ["PATH"] += os.pathsep + poetry_path
        
        # 再次检查 Poetry 是否可用
        if not check_poetry_installed():
            print("Poetry 安装后仍无法使用。请手动将 Poetry 添加到 PATH 环境变量，然后重新运行此脚本。")
            print(f"Poetry 可能安装在: {poetry_path}")
            sys.exit(1)
    else:
        print("检测到 Poetry 已安装")
    
    # 设置项目
    setup_project()
    
    print("\n=== 安装完成 ===")
    print("您现在可以使用以下命令激活虚拟环境并运行项目:")
    print("  poetry shell")
    print("  python scripts/start_all.py")
    print("\n或者直接运行:")
    print("  poetry run python scripts/start_all.py")

if __name__ == "__main__":
    main()
