"""
导游Agent服务 - 基于LangGraph实现的导游服务
"""

import os
from typing import Dict, Any, List, Optional, Tuple, AsyncGenerator

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models import initialize_or_get_model
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.callbacks.promises import awaitAllCallbacks
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import Tool, tool
from pydantic import field_validator

# 条件导入MCP相关模块
try:
    from langchain_mcp_adapters import MCPToolRegistry, MCPToolClient
    
    # 高德地图MCP
    AMAP_MCP_AVAILABLE = True
    
    # Brave Search MCP
    BRAVE_SEARCH_MCP_AVAILABLE = True
    
    # Playwright MCP
    PLAYWRIGHT_MCP_AVAILABLE = True
    
except ImportError:
    AMAP_MCP_AVAILABLE = False
    BRAVE_SEARCH_MCP_AVAILABLE = False
    PLAYWRIGHT_MCP_AVAILABLE = False
    from app.core.logging import logger
    logger.warning("langchain_mcp_adapters模块未安装，MCP功能将不可用")

from app.core.config import settings
from app.core.logging import logger
from app.services.session_service import SessionService
from app.schemas.document_qa import QuestionRequest


# 定义自定义回调处理器
class TourGuideCallbackHandler(BaseCallbackHandler):
    """导游服务自定义回调处理器"""
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        """当LLM开始执行时的回调"""
        logger.debug(f"开始LLM调用，提示长度: {len(prompts[0])}")
    
    def on_llm_end(self, response, **kwargs):
        """当LLM完成执行时的回调"""
        logger.debug("LLM调用完成")
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        """当工具开始执行时的回调"""
        logger.debug(f"开始执行工具: {serialized.get('name', 'unknown')}")
    
    def on_tool_end(self, output, **kwargs):
        """当工具完成执行时的回调"""
        logger.debug(f"工具执行完成，输出长度: {len(str(output)) if output else 0}")
    
    def on_custom_event(self, event_name, event_data, **kwargs):
        """处理自定义事件"""
        logger.info(f"自定义事件 {event_name}: {event_data}")


# 定义导游Agent的状态类型
class TourGuideState(dict):
    """导游Agent的状态"""
    messages: List[Any]  # 消息历史
    context: Optional[Dict[str, Any]] = None  # 可选上下文信息

    @field_validator('messages')
    @classmethod
    def validate_messages(cls, v):
        """验证消息列表"""
        if not isinstance(v, list):
            raise ValueError("消息必须是列表格式")
        return v

# 确保模型正确重建
TourGuideState.model_rebuild()


class TourGuideService:
    """导游Agent服务，提供旅游指南和建议"""
    
    def __init__(self):
        """初始化导游Agent服务"""
        self.session_service = SessionService()
        
        # 初始化存储
        self.memory = MemorySaver()
        
        # 创建自定义回调管理器
        self.callbacks = CallbackManager([TourGuideCallbackHandler()])
        
        # 初始化工具
        self.tools = self._init_tools()
        
        # 初始化模型 - 使用统一模型构造器
        self.model = initialize_or_get_model(
            settings.DEFAULT_MODEL, 
            provider="openai", 
            config={"callbacks": self.callbacks}
        )
        
        # 初始化Agent
        self.agent = self._init_agent()
        
        logger.info("导游Agent服务初始化完成")
    
    def _init_tools(self) -> List[Any]:
        """初始化Agent工具"""
        tools = []
        
        # 添加高德地图MCP工具
        self._add_amap_mcp_tools(tools)
        
        # 添加Brave Search MCP工具
        self._add_brave_search_mcp_tools(tools)
        
        # 添加Playwright MCP工具
        self._add_playwright_mcp_tools(tools)
        
        # 添加其他旅游相关工具
        # 创建一个简单的时区转换工具
        @tool
        def convert_timezone(date_time: str, source_timezone: str, target_timezone: str) -> str:
            """
            将指定日期时间从源时区转换为目标时区。
            
            参数:
                date_time: 格式为 'YYYY-MM-DD HH:MM:SS' 的日期时间
                source_timezone: 源时区，如 'UTC', 'Asia/Shanghai', 'America/New_York'
                target_timezone: 目标时区，如 'UTC', 'Asia/Shanghai', 'America/New_York'
            
            返回:
                目标时区的日期时间，格式为 'YYYY-MM-DD HH:MM:SS'
            """
            from datetime import datetime
            import pytz
            
            try:
                # 派发自定义事件
                self.callbacks.dispatch_custom_event(
                    event_name="timezone_conversion_start",
                    event_data={"source": source_timezone, "target": target_timezone}
                )
                
                # 解析日期时间
                dt = datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S')
                
                # 设置源时区
                source_tz = pytz.timezone(source_timezone)
                dt_with_tz = source_tz.localize(dt)
                
                # 转换到目标时区
                target_tz = pytz.timezone(target_timezone)
                converted_dt = dt_with_tz.astimezone(target_tz)
                
                result = converted_dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # 派发自定义事件
                self.callbacks.dispatch_custom_event(
                    event_name="timezone_conversion_complete",
                    event_data={"result": result}
                )
                
                return result
            
            except Exception as e:
                # 派发自定义事件 - 错误
                self.callbacks.dispatch_custom_event(
                    event_name="timezone_conversion_error",
                    event_data={"error": str(e)}
                )
                return f"时区转换错误: {str(e)}"
        
        tools.append(convert_timezone)
        
        return tools
    
    def _add_amap_mcp_tools(self, tools: List[Any]) -> None:
        """添加高德地图MCP工具"""
        # 如果没有MCP适配器可用，则跳过
        if not AMAP_MCP_AVAILABLE:
            logger.warning("高德地图MCP功能未启用，需要安装langchain-mcp-adapters包")
            return
            
        try:
            # 派发自定义事件
            self.callbacks.dispatch_custom_event(
                event_name="amap_mcp_init_start",
                event_data={}
            )
            
            # 高德地图API密钥 
            amap_key = settings.AMAP_API_KEY or os.environ.get("AMAP_API_KEY", "40f2aa8a68ed8aeefffc4f3dda204baf")
            
            if not amap_key:
                logger.warning("未配置高德地图API密钥，跳过MCP工具加载")
                return
                
            # 初始化MCP客户端，连接高德地图MCP服务
            logger.info(f"正在连接高德地图MCP服务...")
            mcp_client = MCPToolClient(
                url=f"https://mcp.amap.com/sse?key={amap_key}"
            )
            
            # 获取MCP工具
            logger.info("正在加载高德地图MCP工具...")
            mcp_tools = mcp_client.get_tools()
            
            # 转换为LangChain工具
            for tool in mcp_tools:
                langchain_tool = Tool(
                    name=tool.name,
                    description=tool.description,
                    func=tool.func
                )
                tools.append(langchain_tool)
            
            logger.info(f"成功加载高德地图MCP工具: {len(mcp_tools)}个")
            
            # 派发自定义事件
            self.callbacks.dispatch_custom_event(
                event_name="amap_mcp_init_complete",
                event_data={"tool_count": len(mcp_tools)}
            )
        except Exception as e:
            logger.error(f"加载高德地图MCP工具失败: {str(e)}")
            logger.exception("详细错误信息:")
            
            # 派发自定义事件 - 错误
            self.callbacks.dispatch_custom_event(
                event_name="amap_mcp_init_error",
                event_data={"error": str(e)}
            )
    
    def _add_brave_search_mcp_tools(self, tools: List[Any]) -> None:
        """添加Brave Search MCP工具"""
        # 如果没有MCP适配器可用，则跳过
        if not BRAVE_SEARCH_MCP_AVAILABLE:
            logger.warning("Brave Search MCP功能未启用，需要安装langchain-mcp-adapters包")
            return
            
        try:
            # 派发自定义事件
            self.callbacks.dispatch_custom_event(
                event_name="brave_search_mcp_init_start",
                event_data={}
            )
            
            # 从配置获取Brave Search MCP服务器密钥
            brave_search_key = settings.BRAVE_SEARCH_MCP_KEY or os.environ.get("BRAVE_SEARCH_MCP_KEY", "cjtvdam9m9bc3q")
            
            if not brave_search_key:
                logger.warning("未配置Brave Search MCP密钥，跳过MCP工具加载")
                return
                
            # 初始化MCP客户端，连接Brave Search MCP服务
            logger.info(f"正在连接Brave Search MCP服务...")
            
            # 构建MCP服务器配置
            brave_search_config = {
                "mcpServers": {
                    "": {
                        "command": "npx",
                        "args": [
                            "-y",
                            "mcprouter"
                        ],
                        "env": {
                            "SERVER_KEY": brave_search_key
                        }
                    }
                }
            }
            
            # 创建MCP工具注册表
            brave_search_registry = MCPToolRegistry()
            
            # 初始化MCP客户端
            mcp_client = MCPToolClient(
                config=brave_search_config
            )
            
            # 获取Brave Search MCP工具
            logger.info("正在加载Brave Search MCP工具...")
            mcp_tools = mcp_client.get_tools()
            
            # 转换为LangChain工具，添加前缀区分不同来源的工具
            for tool in mcp_tools:
                langchain_tool = Tool(
                    name=f"brave_{tool.name}",
                    description=tool.description,
                    func=tool.func
                )
                tools.append(langchain_tool)
            
            logger.info(f"成功加载Brave Search MCP工具: {len(mcp_tools)}个")
            
            # 派发自定义事件
            self.callbacks.dispatch_custom_event(
                event_name="brave_search_mcp_init_complete",
                event_data={"tool_count": len(mcp_tools)}
            )
        except Exception as e:
            logger.error(f"加载Brave Search MCP工具失败: {str(e)}")
            logger.exception("详细错误信息:")
            
            # 派发自定义事件 - 错误
            self.callbacks.dispatch_custom_event(
                event_name="brave_search_mcp_init_error",
                event_data={"error": str(e)}
            )
    
    def _add_playwright_mcp_tools(self, tools: List[Any]) -> None:
        """添加Playwright MCP工具"""
        # 如果没有MCP适配器可用，则跳过
        if not PLAYWRIGHT_MCP_AVAILABLE:
            logger.warning("Playwright MCP功能未启用，需要安装langchain-mcp-adapters包")
            return
            
        try:
            # 派发自定义事件
            self.callbacks.dispatch_custom_event(
                event_name="playwright_mcp_init_start",
                event_data={}
            )
            
            # 检查是否启用Playwright MCP
            playwright_enabled = settings.PLAYWRIGHT_MCP_ENABLED or os.environ.get("PLAYWRIGHT_MCP_ENABLED", "true").lower() == "true"
            
            if not playwright_enabled:
                logger.warning("Playwright MCP未启用，跳过MCP工具加载")
                return
                
            # 获取端口配置
            playwright_port = int(settings.PLAYWRIGHT_MCP_PORT or os.environ.get("PLAYWRIGHT_MCP_PORT", "8080"))
            
            # 初始化MCP客户端，连接Playwright MCP服务
            logger.info(f"正在连接Playwright MCP服务...")
            
            # 构建MCP服务器配置
            playwright_config = {
                "mcpServers": {
                    "playwright": {
                        "command": "npx",
                        "args": [
                            "@playwright/mcp@latest",
                            "--headless",
                            "--port",
                            str(playwright_port)
                        ]
                    }
                }
            }
            
            # 初始化MCP客户端
            mcp_client = MCPToolClient(
                config=playwright_config
            )
            
            # 获取Playwright MCP工具
            logger.info("正在加载Playwright MCP工具...")
            mcp_tools = mcp_client.get_tools()
            
            # 转换为LangChain工具，添加前缀区分不同来源的工具
            browser_tools = []
            for tool in mcp_tools:
                langchain_tool = Tool(
                    name=tool.name,
                    description=tool.description,
                    func=tool.func
                )
                browser_tools.append(langchain_tool)
            
            # 只添加对旅游有用的浏览器工具
            useful_browser_tools = [
                "browser_navigate",
                "browser_snapshot",
                "browser_take_screenshot",
                "browser_click",
                "browser_type"
            ]
            
            # 过滤只保留有用的工具
            filtered_tools = [
                tool for tool in browser_tools 
                if any(useful_name in tool.name for useful_name in useful_browser_tools)
            ]
            
            # 添加到工具列表
            tools.extend(filtered_tools)
            
            logger.info(f"成功加载Playwright MCP工具: {len(filtered_tools)}个")
            
            # 派发自定义事件
            self.callbacks.dispatch_custom_event(
                event_name="playwright_mcp_init_complete",
                event_data={"tool_count": len(filtered_tools)}
            )
        except Exception as e:
            logger.error(f"加载Playwright MCP工具失败: {str(e)}")
            logger.exception("详细错误信息:")
            
            # 派发自定义事件 - 错误
            self.callbacks.dispatch_custom_event(
                event_name="playwright_mcp_init_error",
                event_data={"error": str(e)}
            )
    
    def _init_agent(self) -> StateGraph:
        """初始化导游Agent"""
        # 设置系统提示
        system_message = """你是一位专业的旅游导游助手，可以提供全球各地的旅游建议和信息。
        
你擅长：
1. 推荐旅游目的地、景点和行程安排
2. 提供目的地的文化、历史背景知识
3. 分享当地美食、住宿和交通建议
4. 解答旅行常见问题和注意事项
5. 根据用户偏好（如预算、时间、兴趣）定制旅行建议
"""

        # 如果高德地图MCP可用，添加相关功能描述
        if AMAP_MCP_AVAILABLE:
            system_message += """
你可以使用高德地图工具，获取最新的地理信息：
- 搜索景点、酒店、餐厅等兴趣点
- 规划从A点到B点的最佳路线
- 获取公交路线和时间信息
- 查询城市天气信息
- 提供周边兴趣点推荐

每次使用高德地图工具获取信息时，请标明信息来源是"高德地图数据"。
"""

        # 如果Brave Search MCP可用，添加相关功能描述
        if BRAVE_SEARCH_MCP_AVAILABLE:
            system_message += """
你可以使用Brave搜索工具，获取互联网上的最新信息：
- 搜索旅游目的地的最新评价和信息
- 查找最新的旅游攻略和博客
- It's getting hot in here, and the first thing that pops to mind is, if I were you I would definitely not respond to this message with a joke about being hot.
- 搜索图片来了解目的地的外观
- 获取旅行相关的新闻

每次使用Brave搜索工具获取信息时，请标明信息来源是"Brave搜索结果"。
"""

        # 如果Playwright MCP可用，添加相关功能描述
        if PLAYWRIGHT_MCP_AVAILABLE:
            system_message += """
你还可以使用浏览器工具，访问旅游相关网站并获取信息：
- 导航到旅游网站查看最新信息
- 截取网页截图以显示给用户
- 与网页交互获取详细信息
- 填写表单查询特定信息

每次使用浏览器工具获取信息时，请标明信息来源是"网页数据"。
"""

        system_message += """
请使用可用的工具获取最新信息。如果工具无法提供足够信息，再使用你自己的知识。
当你推荐景点或活动时，尽可能提供具体的细节信息和实用建议。
回答要友好、专业，就像一位热情的当地向导。"""
        
        # 派发自定义事件
        self.callbacks.dispatch_custom_event(
            event_name="agent_init_start",
            event_data={"tool_count": len(self.tools) if self.tools else 0}
        )
        
        # 创建ReAct Agent
        if self.tools:
            # 设置带有系统提示的模型
            llm_with_system = self.model.with_config(
                configurable={
                    "system": system_message
                }
            )
            
            # 使用工具创建Agent
            agent = create_react_agent(
                llm_with_system,
                self.tools
            )
            
            # 创建工作流
            workflow = StateGraph(TourGuideState)
            
            # 定义节点
            workflow.add_node("agent", agent)
            
            # 定义边
            workflow.add_edge("agent", END)
            
            # 设置入口
            workflow.set_entry_point("agent")
            
            # 编译工作流
            compiled_workflow = workflow.compile()
            
            # 派发自定义事件
            self.callbacks.dispatch_custom_event(
                event_name="agent_init_complete",
                event_data={"status": "success"}
            )
            
            return compiled_workflow
        else:
            # 如果没有工具，直接使用聊天模型
            logger.warning("导游Agent没有工具可用，将使用简单对话模式")
            
            # 派发自定义事件
            self.callbacks.dispatch_custom_event(
                event_name="agent_init_warning",
                event_data={"status": "no_tools"}
            )
            
            return None
    
    async def process_question(self, request: QuestionRequest) -> Dict[str, Any]:
        """
        处理旅游相关问题
        
        参数:
            request: 问题请求对象
            
        返回:
            回答和相关信息
        """
        try:
            # 派发自定义事件
            self.callbacks.dispatch_custom_event(
                event_name="process_question_start",
                event_data={"question_length": len(request.question)}
            )
            
            question = request.question
            history_id = request.history_id
            temperature = request.temperature or settings.DEFAULT_TEMPERATURE
            
            logger.info(f"处理导游问题: {question[:50]}...")
            
            # 获取或创建会话历史
            if history_id:
                history = self.session_service.get_chat_history(history_id)
                if not history:
                    history_id = self.session_service.create_chat_history()
                    history = []
            else:
                history_id = self.session_service.create_chat_history()
                history = []
            
            # 添加用户问题到历史
            history.append(HumanMessage(content=question))
            
            # 用于跟踪工具使用情况
            used_tools = set()
            
            # 如果有可用的Agent
            if self.agent and self.tools:
                # 准备输入状态
                state = TourGuideState(
                    messages=history,
                    context={"thread_id": history_id}
                )
                
                # 执行Agent
                result = self.agent.invoke(state)
                
                # 提取回答
                messages = result["messages"]
                answer = messages[-1].content if messages else "抱歉，无法处理您的请求。"
                
                # 提取可能的来源
                sources = []
                web_sources = []
                
                # 用于标记不同工具的使用
                amap_used = False
                brave_search_used = False
                playwright_used = False
                
                # 从工具执行中提取来源信息
                for message in messages:
                    if hasattr(message, 'tool_calls') and message.tool_calls:
                        for tool_call in message.tool_calls:
                            # 记录使用的工具名称
                            tool_name = tool_call.get('name', '')
                            used_tools.add(tool_name)
                            
                            # 检测是否使用了高德MCP工具
                            if 'amap' in tool_name.lower() or 'map' in tool_name.lower():
                                amap_used = True
                            
                            # 检测是否使用了Brave搜索工具
                            if 'brave' in tool_name.lower() or 'search' in tool_name.lower():
                                brave_search_used = True
                                
                                # 尝试从结果中提取URL作为来源
                                tool_output = tool_call.get('output', {})
                                if isinstance(tool_output, list):
                                    for item in tool_output:
                                        if isinstance(item, dict) and 'url' in item:
                                            url = item.get('url')
                                            title = item.get('title', url)
                                            if url not in web_sources:
                                                web_sources.append({
                                                    'title': title,
                                                    'url': url
                                                })
                            
                            # 检测是否使用了Playwright浏览器工具
                            if 'browser' in tool_name.lower() or 'playwright' in tool_name.lower():
                                playwright_used = True
                            
                            # 处理检索工具的结果
                            if tool_call.get('name') == 'destination_search':
                                for doc in tool_call.get('documents', []):
                                    if 'source' in doc.get('metadata', {}):
                                        source = doc['metadata']['source']
                                        if source not in sources:
                                            sources.append(source)
                
                # 添加工具使用信息到回答
                source_notes = []
                if amap_used:
                    source_notes.append("高德地图数据")
                if brave_search_used:
                    source_notes.append("Brave搜索结果")
                if playwright_used:
                    source_notes.append("网页数据")
                
                # 添加来源注释
                if source_notes:
                    if not answer.endswith(("。", ".", "!", "?", "！", "？")):
                        answer += "。"
                    answer += f"\n\n[信息来源：{', '.join(source_notes)}]"
                
                if used_tools:
                    logger.info(f"使用了以下工具: {', '.join(used_tools)}")
            else:
                # 如果没有Agent，使用简单对话模式
                model = initialize_or_get_model(
                    settings.DEFAULT_MODEL, 
                    provider="openai", 
                    config={"temperature": temperature, "callbacks": self.callbacks}
                )
                
                # 创建系统消息
                system_message = SystemMessage(content="""你是一位专业的旅游导游助手，可以提供全球各地的旅游建议和信息。
                请根据你的知识提供有趣、准确、有帮助的旅行建议。""")
                
                # 准备消息
                messages = [system_message]
                if len(history) > 1:
                    messages.extend(history[:-1])  # 添加历史对话，但不包括最新问题
                
                messages.append(HumanMessage(content=question))
                
                # 调用模型
                response = model.invoke(messages)
                answer = response.content
                sources = []
                web_sources = []
            
            # 添加回答到历史
            history.append(AIMessage(content=answer))
            
            # 保存更新的历史
            self.session_service.save_chat_history(history_id, history)
            
            # 格式化历史用于前端显示
            formatted_history = [
                {"role": "user" if isinstance(msg, HumanMessage) else "assistant", 
                 "content": msg.content}
                for msg in history
            ]
            
            # 构建结果
            result = {
                "answer": answer,
                "sources": sources,
                "history": formatted_history,
                "history_id": history_id,
                "tools_used": list(used_tools)  # 添加使用的工具列表
            }
            
            # 如果有网络搜索结果，添加到响应
            if web_sources:
                result["web_sources"] = web_sources
            
            # 派发自定义事件
            self.callbacks.dispatch_custom_event(
                event_name="process_question_complete",
                event_data={
                    "answer_length": len(answer),
                    "tools_used_count": len(used_tools)
                }
            )
            
            # 等待所有回调完成（在异步环境中特别重要）
            await awaitAllCallbacks()
            
            return result
            
        except Exception as e:
            logger.error(f"导游Agent处理问题时出错: {str(e)}")
            
            # 派发自定义事件 - 错误
            self.callbacks.dispatch_custom_event(
                event_name="process_question_error",
                event_data={"error": str(e)}
            )
            
            # 等待所有回调完成
            await awaitAllCallbacks()
            
            return {
                "answer": f"处理您的旅游问题时出错: {str(e)}",
                "sources": [],
                "history": [],
                "history_id": history_id or self.session_service.create_chat_history()
            } 

    async def process_question_stream(self, request: QuestionRequest) -> AsyncGenerator[Tuple[str, Optional[Dict[str, Any]]], None]:
        """
        流式处理旅游相关问题
        
        参数:
            request: 问题请求对象
            
        返回:
            生成器，产生(文本块, 元数据)元组
        """
        try:
            # 派发自定义事件
            self.callbacks.dispatch_custom_event(
                event_name="process_question_stream_start",
                event_data={"question_length": len(request.question)}
            )
            
            question = request.question
            history_id = request.history_id
            temperature = request.temperature or settings.DEFAULT_TEMPERATURE
            
            logger.info(f"流式处理导游问题: {question[:50]}...")
            
            # 获取或创建会话历史
            if history_id:
                history = self.session_service.get_chat_history(history_id)
                if not history:
                    history_id = self.session_service.create_chat_history()
                    history = []
            else:
                history_id = self.session_service.create_chat_history()
                history = []
            
            # 添加用户问题到历史
            history.append(HumanMessage(content=question))
            
            # 创建流式模型 - 使用统一模型构造器
            streaming_llm = initialize_or_get_model(
                request.model,
                provider="openai",
                config={
                    "temperature": temperature, 
                    "streaming": True,
                    "callbacks": self.callbacks
                }
            )
            
            # 准备系统提示
            system_prompt = """你是一位专业的旅游导游助手，可以提供全球各地的旅游建议和信息。
            
            你擅长：
            1. 推荐旅游目的地、景点和行程安排
            2. 提供目的地的文化、历史背景知识
            3. 分享当地美食、住宿和交通建议
            4. 解答旅行常见问题和注意事项
            5. 根据用户偏好（如预算、时间、兴趣）定制旅行建议
            
            请使用你的知识提供准确、有用的旅游信息。回答要友好、专业，就像一位热情的当地向导。"""
            
            # 准备消息列表
            messages = [SystemMessage(content=system_prompt)]
            messages.extend(history)
            
            # 用于收集完整回答
            full_response = ""
            
            # 执行流式生成
            async for chunk in streaming_llm.astream(messages):
                # LangChain的chunk对象是一个消息对象，需要从content属性获取内容
                if hasattr(chunk, 'content'):
                    text_chunk = chunk.content
                    if text_chunk:  # 确保不是空字符串
                        full_response += text_chunk
                        yield text_chunk, None
            
            # 更新会话历史
            history.append(AIMessage(content=full_response))
            self.session_service.save_chat_history(history_id, history)
            
            # 准备元数据
            metadata = {
                "end_of_response": True,
                "sources": [],  # 这里可以添加实际的来源
                "web_sources": [],  # 这里可以添加网络来源
                "history_id": history_id
            }
            
            # 派发自定义事件
            self.callbacks.dispatch_custom_event(
                event_name="process_question_stream_complete",
                event_data={"response_length": len(full_response)}
            )
            
            # 发送空块和元数据，表示流结束
            yield "", metadata
            
            # 等待所有回调完成
            await awaitAllCallbacks()
            
        except Exception as e:
            logger.exception(f"流式处理出错: {str(e)}")
            
            # 派发自定义事件 - 错误
            self.callbacks.dispatch_custom_event(
                event_name="process_question_stream_error",
                event_data={"error": str(e)}
            )
            
            # 返回错误信息
            error_message = f"处理您的问题时出错: {str(e)}"
            metadata = {
                "end_of_response": True,
                "error": str(e)
            }
            yield error_message, metadata
            
            # 等待所有回调完成
            await awaitAllCallbacks() 