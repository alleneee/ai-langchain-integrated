"""
文档处理功能测试脚本
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.services.document_processing_service import DocumentProcessingService
from src.factories.extended_document_loader_factory import ExtendedDocumentLoaderFactory

class TestDocumentProcessing(unittest.TestCase):
    """文档处理功能测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.service = DocumentProcessingService()
        
        # 创建临时测试文件
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_files = {}
        
        # 创建测试文本文件
        text_path = os.path.join(self.temp_dir.name, "test.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write("这是一个测试文本文件。\n包含多行内容。\n用于测试文档加载功能。")
        self.test_files["txt"] = text_path
        
        # 创建测试Markdown文件
        md_path = os.path.join(self.temp_dir.name, "test.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# 测试标题\n\n这是一个测试Markdown文件。\n\n## 二级标题\n\n- 列表项1\n- 列表项2")
        self.test_files["md"] = md_path
        
        # 创建测试JSON文件
        json_path = os.path.join(self.temp_dir.name, "test.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write('{"name": "测试", "description": "这是一个测试JSON文件", "items": [1, 2, 3]}')
        self.test_files["json"] = json_path
    
    def tearDown(self):
        """测试后清理"""
        self.temp_dir.cleanup()
    
    def test_supported_formats(self):
        """测试获取支持的文档格式"""
        formats = self.service.get_supported_formats()
        self.assertIsInstance(formats, list)
        self.assertGreater(len(formats), 0)
        
        # 检查常见格式是否被支持
        common_formats = ['.txt', '.pdf', '.docx', '.md', '.json', '.csv', '.html']
        for fmt in common_formats:
            self.assertIn(fmt, formats)
    
    def test_load_text_document(self):
        """测试加载文本文档"""
        documents = self.service.load_document(self.test_files["txt"])
        self.assertGreater(len(documents), 0)
        self.assertIn("这是一个测试文本文件", documents[0].page_content)
    
    def test_load_markdown_document(self):
        """测试加载Markdown文档"""
        documents = self.service.load_document(self.test_files["md"])
        self.assertGreater(len(documents), 0)
        self.assertIn("测试标题", documents[0].page_content)
    
    def test_load_json_document(self):
        """测试加载JSON文档"""
        documents = self.service.load_document(self.test_files["json"])
        self.assertGreater(len(documents), 0)
        self.assertIn("测试", documents[0].page_content)
    
    def test_load_and_split_document(self):
        """测试加载并分割文档"""
        documents = self.service.load_and_split_document(self.test_files["txt"])
        self.assertGreater(len(documents), 0)
        
        # 检查分割是否生效
        original_docs = self.service.load_document(self.test_files["txt"])
        # 如果文本足够长，分割后的文档数应该大于或等于原始文档数
        self.assertGreaterEqual(len(documents), len(original_docs))
    
    def test_factory_create_from_file_path(self):
        """测试工厂创建加载器"""
        loader = ExtendedDocumentLoaderFactory.create_from_file_path(self.test_files["txt"])
        self.assertIsNotNone(loader)
        
        documents = loader.load()
        self.assertGreater(len(documents), 0)
        self.assertIn("这是一个测试文本文件", documents[0].page_content)

if __name__ == "__main__":
    unittest.main()
