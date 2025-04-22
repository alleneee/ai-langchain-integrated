"""
文档处理示例脚本

演示如何使用扩展的文档处理功能
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.services.document_processing_service import DocumentProcessingService
from src.factories.extended_document_loader_factory import ExtendedDocumentLoaderFactory

def main():
    """主函数"""
    # 创建文档处理服务
    service = DocumentProcessingService()
    
    # 获取支持的文档格式
    formats = service.get_supported_formats()
    print(f"支持的文档格式: {formats}")
    
    # 示例1: 加载本地文本文件
    example_file = "examples/example.txt"
    
    # 如果示例文件不存在，创建一个
    if not os.path.exists(example_file):
        os.makedirs(os.path.dirname(example_file), exist_ok=True)
        with open(example_file, "w", encoding="utf-8") as f:
            f.write("这是一个示例文本文件。\n用于演示文档处理功能。\n支持多种文档格式。")
    
    print(f"\n加载文本文件: {example_file}")
    documents = service.load_document(example_file)
    print(f"加载了 {len(documents)} 个文档")
    for i, doc in enumerate(documents):
        print(f"文档 {i+1}:")
        print(f"内容: {doc.page_content[:100]}...")
        print(f"元数据: {doc.metadata}")
    
    # 示例2: 加载并分割文档
    print(f"\n加载并分割文档: {example_file}")
    split_documents = service.load_and_split_document(example_file)
    print(f"分割后得到 {len(split_documents)} 个文档片段")
    for i, doc in enumerate(split_documents):
        print(f"片段 {i+1}:")
        print(f"内容: {doc.page_content}")
        print(f"元数据: {doc.metadata}")
    
    # 示例3: 加载网页内容
    url = "https://www.example.com"
    print(f"\n加载网页内容: {url}")
    try:
        web_documents = service.load_document(url)
        print(f"加载了 {len(web_documents)} 个文档")
        print(f"内容预览: {web_documents[0].page_content[:200]}...")
    except Exception as e:
        print(f"加载网页失败: {str(e)}")
    
    # 示例4: 使用工厂直接创建加载器
    print("\n使用工厂直接创建加载器")
    loader = ExtendedDocumentLoaderFactory.create_from_file_path(example_file)
    factory_documents = loader.load()
    print(f"加载了 {len(factory_documents)} 个文档")
    print(f"内容预览: {factory_documents[0].page_content[:100]}...")

if __name__ == "__main__":
    main()
