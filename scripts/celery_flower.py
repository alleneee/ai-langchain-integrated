"""
Celery Flower 监控启动脚本
"""

import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flower.command import FlowerCommand
from src.celery_app.celery_app import app

if __name__ == '__main__':
    flower = FlowerCommand()
    flower.run_from_argv(['flower', '--port=5555', '--broker=redis://localhost:6379/0'])
