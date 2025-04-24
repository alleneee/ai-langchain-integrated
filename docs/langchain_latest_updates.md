# LangChain 最新更新说明

本文档记录了我们最近对项目中LangChain集成的优化，以适应LangChain最新版本的API变化。

## 主要更新内容

### 1. 参数名称更新

OpenAI等集成模块的参数名称已更新：

- `openai_api_key` → `api_key`
- `openai_api_base` → `base_url`

我们已在所有LLM策略实现中更新了这些参数名称。

### 2. 工具调用优化

为了支持LangChain最新的工具调用模式，我们添加了以下功能：

- 在`OpenAILangchainStrategy`中增加了`chat_with_tools`方法
- 使用更简洁的`bind_tools`模式绑定工具到模型
- 优化了工具调用结果的处理方式

新方法示例：

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

### 3. 消息处理优化

增强了消息处理能力，特别是对工具调用和工具响应的支持：

- 支持处理`ToolMessage`类型
- 完善了对消息中工具调用属性的处理
- 更新了消息转换逻辑

### 4. RAG实现优化

使用LangChain最新的链构建方式重构了检索增强生成(RAG)实现：

- 使用函数管道风格（LCEL - LangChain Expression Language）构建检索链
- 优化了文档格式化方法
- 添加了对文档来源的支持
- 提供了包含和不包含来源信息两种模式

新的RAG实现示例：

```python
def create_retrieval_chain(
    self,
    llm: BaseLanguageModel,
    retriever: BaseRetriever,
    prompt_template: Optional[str] = None,
    include_sources: bool = False,
    **kwargs
):
    # 使用默认或自定义提示模板
    if prompt_template:
        prompt = ChatPromptTemplate.from_template(prompt_template)
    else:
        # 使用更新后的提示模板格式
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个有帮助的助手，使用以下上下文来回答用户的问题。"
                      "如果你在上下文中找不到答案，请说你不知道，不要编造信息。\n\n"
                      "上下文：\n{context}"),
            ("human", "{question}")
        ])
    
    # 使用新的链构建方式
    def format_docs(docs):
        return "\n\n".join(f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs))
            
    # 构建新版RAG链
    retrieval_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # 可选添加来源信息
    if include_sources:
        # 添加源文档逻辑...
        return source_chain
    else:
        return retrieval_chain
```

### 5. 添加了工具处理支持

新增了`process_chat_with_tools`方法，专门处理带工具调用的聊天：

```python
async def process_chat_with_tools(
    self,
    model: BaseChatModel,
    messages: List[Dict],
    tools: List,
    system_message: Optional[str] = None
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    # 消息处理、工具绑定和响应处理逻辑...
```

## 使用新功能

项目中的各模块可以通过以下方式利用这些新功能：

1. 对于需要工具调用的场景，使用`chat_with_tools`方法或`process_chat_with_tools`方法
2. 对于RAG应用，使用新的`create_retrieval_chain`方法，可选是否包含源文档信息
3. 确保消息处理适配了可能的工具调用和工具响应

## 后续工作

1. 更新所有LLM提供商的工具调用支持
2. 完善对各类消息格式的处理
3. 添加工具响应处理流程的单元测试

## 参考资料

- [LangChain工具调用文档](https://python.langchain.com/docs/concepts/tool_calling)
- [LangChain表达式语言(LCEL)文档](https://python.langchain.com/docs/concepts/expression_language/)
- [LangChain消息文档](https://python.langchain.com/docs/concepts/messages/)
