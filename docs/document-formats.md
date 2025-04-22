# 扩展文档格式支持

本文档介绍了如何使用 LangChain 0.3 版本的文档加载器来支持更多文档格式。

## 支持的文档格式

通过集成 LangChain 和 Unstructured 包，我们现在支持以下文档格式：

### 基本文本格式
- 纯文本文件 (.txt)
- Markdown文件 (.md, .markdown)

### 办公文档
- PDF文件 (.pdf)
- Word文档 (.docx, .doc)
- Excel文件 (.xlsx, .xls)
- PowerPoint文件 (.pptx, .ppt)
- OpenDocument文本 (.odt)
- RTF文件 (.rtf)

### 结构化数据
- CSV文件 (.csv)
- JSON文件 (.json)
- JSON Lines文件 (.jsonl)
- XML文件 (.xml)
- HTML文件 (.html, .htm)
- TOML文件 (.toml)
- YAML文件 (.yaml, .yml)

### 电子书
- EPUB文件 (.epub)

### 代码和开发文档
- Python文件 (.py)
- JavaScript文件 (.js)
- Java文件 (.java)
- C/C++文件 (.c, .cpp)
- C#文件 (.cs)
- Go文件 (.go)
- Ruby文件 (.rb)
- reStructuredText文件 (.rst)
- Org-mode文件 (.org)

### 特殊格式
- 电子邮件 (.eml)
- TSV文件 (.tsv)
- WhatsApp聊天记录 (.txt.chat)

### 多媒体和在线内容
- YouTube视频转录
- 网页内容 (URL)
- 图像文件 (通过OCR提取文本)

### 版本控制
- Git仓库

## 安装依赖

要使用所有支持的文档格式，请安装以下依赖：

```bash
pip install -r requirements.txt
```

所有扩展文档处理的依赖已经包含在主 requirements.txt 文件中。

对于某些文档格式，还需要安装系统依赖：

```bash
# Ubuntu/Debian
sudo apt-get install -y libmagic-dev poppler-utils tesseract-ocr qpdf libreoffice pandoc

# macOS
brew install libmagic poppler tesseract qpdf libreoffice pandoc
```

## 使用方法

### 通过API使用

```python
import requests

# 上传文件
files = {'file': open('document.pdf', 'rb')}
data = {'split': True, 'chunk_size': 1000, 'chunk_overlap': 200}
response = requests.post('http://localhost:8000/documents/upload', files=files, data=data)
print(response.json())

# 处理URL或路径
data = {'source': 'https://example.com', 'split': True}
response = requests.post('http://localhost:8000/documents/process', data=data)
print(response.json())

# 获取支持的格式
response = requests.get('http://localhost:8000/documents/formats')
print(response.json())
```

### 通过服务使用

```python
from src.services.document_processing_service import DocumentProcessingService

# 创建服务实例
service = DocumentProcessingService()

# 加载文档
documents = service.load_document('document.pdf')

# 加载并分割文档
split_documents = service.load_and_split_document('document.pdf')

# 获取支持的格式
formats = service.get_supported_formats()
```

## 注意事项

1. 对于某些特殊格式，可能需要额外的配置或依赖。
2. 图像文件的文本提取依赖于OCR质量，可能不总是准确。
3. 对于大型文件，建议使用分块处理以避免内存问题。
4. 某些格式可能需要特定的处理参数，请参考API文档。

## 故障排除

如果遇到文档加载问题，请尝试以下步骤：

1. 确保已安装所有必要的依赖。
2. 检查文件格式是否受支持。
3. 对于复杂格式，尝试使用 `use_unstructured=True` 参数。
4. 查看日志以获取详细的错误信息。

如果问题仍然存在，请提交issue并附上错误日志和文件格式信息。
