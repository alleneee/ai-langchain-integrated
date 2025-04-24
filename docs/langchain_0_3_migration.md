# LangChain 0.3 迁移指南

本文档提供了将 Dify-Connect 项目从 LangChain 0.2 升级到 LangChain 0.3 的说明和注意事项。

## 主要变化

LangChain 0.3 的主要变更包括：

1. **全面升级到 Pydantic 2**: 所有包内部已从 Pydantic 1 升级到 Pydantic 2，不再需要使用兼容层如 `langchain_core.pydantic_v1`。
2. **不再支持 Pydantic 1**: Pydantic 1 于 2024 年 6 月停止维护，不再受支持。
3. **不再支持 Python 3.8**: Python 3.8 将于 2024 年 10 月结束生命周期，不再受支持。
4. **新的工具调用API**: 引入了更简洁的工具绑定与调用方式。
5. **改进的RAG实现**: 使用新的链构建方式和文档格式化方法。
6. **更新的参数名称**: 部分集成包如OpenAI更新了参数命名。

## 升级步骤

### 1. 更新依赖

首先，更新项目中的 LangChain 相关依赖：

```bash
# 使用pip
pip install -U "langchain>=0.3.0,<0.4.0" "langchain-core>=0.3.0,<0.4.0" "langchain-community>=0.3.0,<0.4.0"

# 或使用 poetry
poetry add langchain@^0.3.0 langchain-core@^0.3.0 langchain-community@^0.3.0
```

同时，还需要更新各集成包。LangChain 将更多集成从 `langchain-community` 移动到了独立的包：

```bash
# 安装所需的集成包
pip install "langchain-openai>=0.3.0,<0.4.0" "langchain-anthropic>=0.3.0,<0.4.0" "langchain-google-genai>=2.0.0,<3.0.0"
```

### 2. 迁移 Pydantic 导入

使用项目中提供的迁移脚本替换 Pydantic v1 导入：

```bash
# 先进行模拟运行查看影响
python scripts/migrate_to_pydantic_v2.py src/ --dry-run

# 实际运行更新
python scripts/migrate_to_pydantic_v2.py src/
```

主要的导入更新包括：

- 从 `langchain_core.pydantic_v1` 更新为 `pydantic`
- 从 `langchain.pydantic_v1` 更新为 `pydantic`
- 从 `pydantic.v1` 更新为 `pydantic`

### 3. 更新验证器

Pydantic 2 使用不同的验证器装饰器，主要需要更新：

- `@validator` → `@field_validator`
- `@root_validator` → `@model_validator`

### 4. 添加 model_rebuild()

在继承 LangChain 模型（如 `BaseTool`, `BaseChatModel`, `BaseOutputParser` 等）的子类中，可能需要添加 `model_rebuild()` 调用。示例：

```python
from typing import Optional
from langchain_core.output_parsers import BaseOutputParser

class CustomParser(BaseOutputParser):
    # ...类定义
    pass

# 在类定义之后添加这一行
CustomParser.model_rebuild()
```

### 5. 更新使用特定模型的代码

如果您的代码使用以下API：

- `BaseChatModel.bind_tools`
- `BaseChatModel.with_structured_output`
- `Tool.from_function`
- `StructuredTool.from_function`

确保传入的是 Pydantic 2 对象，而非 Pydantic 1 对象。

### 6. 更新导入路径

某些集成已从 `langchain_community` 移动到独立的包，需要更新导入路径。例如：

```python
# 旧导入
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings

# 新导入
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
```

### 7. 更新OpenAI集成参数名称

OpenAI相关的客户端参数已更新：

```python
# 旧参数名
client = ChatOpenAI(
    openai_api_key="your-key",
    openai_api_base="https://api.example.com",
)

# 新参数名
client = ChatOpenAI(
    api_key="your-key",
    base_url="https://api.example.com",
)
```

### 8. 更新工具调用实现

LangChain 0.3 引入了更简化的工具调用方式：

```python
# 旧方式
tools = [tool1, tool2]
response = llm.generate_with_tools(messages, tools)

# 新方式
tools = [tool1, tool2]
llm_with_tools = llm.bind_tools(tools)
response = llm_with_tools.invoke(messages)
```

我们的实现示例：

```python
@handle_llm_exception
async def chat_with_tools(self, prompt: str, context: List[Dict], tools: List[BaseTool], 
                       model: str, temperature: float, max_tokens: int, **kwargs) -> Dict[str, Any]:
    """使用工具进行聊天，符合LangChain最新工具调用模式。"""
    llm = self._create_chat_client(model, temperature, max_tokens)
    messages = self._prepare_langchain_messages(context, prompt)
    
    # 绑定工具到模型
    llm_with_tools = llm.bind_tools(tools)
    
    # 调用模型生成响应
    response = await llm_with_tools.ainvoke(messages, **kwargs)
    
    return {
        "content": response.content,
        "tool_calls": getattr(response, "tool_calls", None)
    }
```

### 9. 更新RAG实现

LangChain 0.3 改进了检索增强生成(RAG)链的构建方式：

```python
# 旧方式
from langchain.chains import RetrievalQA
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

# 新方式 - 使用LCEL (LangChain Expression Language)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs))

retrieval_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

### 10. 测试应用程序

完成以上更新后，运行应用程序的测试套件，确保所有功能正常运行：

```bash
pytest
```

## 特定集成更新

### Google Gemini

Gemini 集成已从 `langchain_google_community` 更新为 `langchain_google_genai`：

```python
# 旧导入
from langchain_google_community import ChatGoogleGenerativeAI
from langchain_google_community import GoogleGenerativeAIEmbeddings

# 新导入
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
```

### Anthropic

Anthropic 集成已更新到专用包：

```python
# 旧导入
from langchain_community.chat_models import ChatAnthropic

# 新导入
from langchain_anthropic import ChatAnthropic
```

## 消息处理更新

LangChain 0.3 增强了消息处理能力，特别是对工具消息的支持：

```python
# 更新后的消息转换方法示例
def _convert_to_langchain_messages(self, messages: List[Dict]) -> List[BaseMessage]:
    lc_messages = []
    for message in messages:
        role = message.get("role", "user").lower()
        content = message.get("content", "")
        
        # 处理可能的工具调用内容
        tool_calls = message.get("tool_calls", [])
        tool_call_id = message.get("tool_call_id")
        
        if content:
            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant" or role == "ai":
                # 处理可能的工具调用
                if tool_calls:
                    # 创建带有工具调用的AI消息
                    lc_messages.append(AIMessage(content=content, tool_calls=tool_calls))
                else:
                    lc_messages.append(AIMessage(content=content))
            elif role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "tool" and tool_call_id:
                # 添加工具消息支持
                lc_messages.append(ToolMessage(content=content, tool_call_id=tool_call_id))
                
    return lc_messages
```

## 故障排除

### 类型错误

如果遇到类型错误，可能是由于 Pydantic 2 与 Pydantic 1 的差异造成的。检查：

1. 是否使用了已更新的验证器装饰器
2. 是否在子类中调用了 `model_rebuild()`
3. 是否传递了正确版本的 Pydantic 模型给工具或函数

### 导入错误

如果遇到导入错误，确保已安装所有必要的集成包。LangChain 0.3 使用独立的包进行集成，部分功能已从 `langchain-community` 移动。

### 工具调用错误

如果工具调用失败，检查：

1. 工具定义是否使用了 Pydantic 2 模型
2. 是否正确地将工具绑定到模型
3. 消息格式是否正确处理了工具调用和工具响应

## 参考资料

- [LangChain 0.3 官方文档](https://python.langchain.com/docs/versions/v0_3/)
- [Pydantic 2 迁移指南](https://docs.pydantic.dev/2.0/migration/)
- [LangChain GitHub 仓库](https://github.com/langchain-ai/langchain)
- [工具调用文档](https://python.langchain.com/docs/concepts/tool_calling)
- [LCEL 文档](https://python.langchain.com/docs/concepts/expression_language/)
