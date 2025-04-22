"""
异步文档处理示例脚本

演示如何使用异步文档处理功能
"""

import os
import sys
import time
import requests
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 注意: 运行前请确保已安装了所有依赖：
# pip install -r requirements.txt
# 并且 Redis 服务已启动

def main():
    """主函数"""
    # API基础URL
    base_url = "http://localhost:8000"

    # 示例1: 获取支持的文档格式
    print("获取支持的文档格式...")
    response = requests.get(f"{base_url}/documents/async/formats")
    formats = response.json()["formats"]
    print(f"支持的文档格式: {formats}")

    # 示例2: 异步处理URL
    url = "https://www.example.com"
    print(f"\n异步处理URL: {url}")

    # 提交异步处理任务
    data = {'source': url, 'split': True}
    response = requests.post(f"{base_url}/documents/async/process", data=data)

    if response.status_code == 200:
        task_data = response.json()
        task_id = task_data["task_id"]
        print(f"任务已提交，任务ID: {task_id}")

        # 轮询任务状态
        while True:
            status_response = requests.get(f"{base_url}/documents/async/tasks/{task_id}")
            status_data = status_response.json()

            print(f"任务状态: {status_data['status']}")

            if status_data['status'] == 'SUCCESS':
                print("任务完成!")
                result = status_data['result']
                print(f"处理了 {result['document_count']} 个文档")
                print(f"第一个文档内容预览: {result['documents'][0]['page_content'][:100]}...")
                break
            elif status_data['status'] == 'FAILURE':
                print(f"任务失败: {status_data.get('error', '未知错误')}")
                break

            # 等待一段时间再次查询
            time.sleep(1)
    else:
        print(f"提交任务失败: {response.text}")

    # 示例3: 异步处理本地文件
    # 创建示例文件
    example_file = "examples/example.txt"
    if not os.path.exists(example_file):
        os.makedirs(os.path.dirname(example_file), exist_ok=True)
        with open(example_file, "w", encoding="utf-8") as f:
            f.write("这是一个示例文本文件。\n用于演示异步文档处理功能。\n支持多种文档格式。")

    print(f"\n异步处理本地文件: {example_file}")

    # 上传文件进行异步处理
    with open(example_file, 'rb') as f:
        files = {'file': (os.path.basename(example_file), f)}
        data = {'split': False}
        response = requests.post(f"{base_url}/documents/async/upload", files=files, data=data)

    if response.status_code == 200:
        task_data = response.json()
        task_id = task_data["task_id"]
        print(f"任务已提交，任务ID: {task_id}")

        # 轮询任务状态
        while True:
            status_response = requests.get(f"{base_url}/documents/async/tasks/{task_id}")
            status_data = status_response.json()

            print(f"任务状态: {status_data['status']}")

            if status_data['status'] == 'SUCCESS':
                print("任务完成!")
                result = status_data['result']
                print(f"处理了 {result['document_count']} 个文档")
                print(f"第一个文档内容预览: {result['documents'][0]['page_content'][:100]}...")
                break
            elif status_data['status'] == 'FAILURE':
                print(f"任务失败: {status_data.get('error', '未知错误')}")
                break

            # 等待一段时间再次查询
            time.sleep(1)
    else:
        print(f"提交任务失败: {response.text}")

if __name__ == "__main__":
    main()
