"""
Google Gemini LLM提供商适配器

实现了Google Gemini API的调用与适配，使用langchain-google-community库
"""

import json
import tiktoken
from typing import Dict, List, Any, Optional, AsyncGenerator
from src.services.llm.base import BaseLLMProvider
from src.core.exceptions import (
    LLMProviderException, LLMProviderAuthException,
    LLMProviderModelNotFoundException
)
import asyncio

class GeminiProvider(BaseLLMProvider):
    """Google Gemini API适配器"""
    
    def __init__(self, api_key: str = None, api_base: str = None, **kwargs):
        """初始化Gemini适配器
        
        Args:
            api_key: Google API密钥
            api_base: 自定义API基础URL (通常不需要)
            **kwargs: 其他参数
        """
        super().__init__(api_key, api_base, **kwargs)
        self._setup_client()
    
    def _setup_client(self):
        """设置Gemini客户端"""
        try:
            self._validate_api_key()
            
            # 尝试导入langchain-google-community
            try:
                from langchain_google_community import ChatGoogleGenerativeAI
                self.chat_model_cls = ChatGoogleGenerativeAI
                self.use_langchain = True
            except ImportError:
                # 回退到使用Google GenerativeAI SDK
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.genai = genai
                self.use_langchain = False
        except Exception as e:
            raise self._format_exception(e)
    
    async def generate_text(self, prompt: str, context: List[Dict], model: str,
                         temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """生成文本响应
        
        Args:
            prompt: 提示文本
            context: 上下文消息列表
            model: 模型名称，如gemini-pro或gemini-ultra
            temperature: 温度参数
            max_tokens: 最大生成令牌数
            
        Returns:
            str: 生成的文本
            
        Raises:
            LLMProviderException: Gemini API调用异常
        """
        try:
            if self.use_langchain:
                # 使用LangChain集成
                from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
                
                # 转换为LangChain消息
                messages = []
                for msg in context:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    
                    if role == "user":
                        messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        messages.append(AIMessage(content=content))
                    elif role == "system":
                        messages.append(SystemMessage(content=content))
                
                # 添加当前提示
                if prompt:
                    messages.append(HumanMessage(content=prompt))
                
                # 创建LangChain模型
                chat_model = self.chat_model_cls(
                    model=model,
                    google_api_key=self.api_key,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    convert_system_message_to_human=True  # Gemini不直接支持系统消息
                )
                
                # 调用模型
                response = await asyncio.to_thread(chat_model.invoke, messages)
                
                # 提取响应内容
                if hasattr(response, "content"):
                    return response.content
                else:
                    return str(response)
            else:
                # 使用Google GenerativeAI SDK
                
                # 转换为Gemini格式的聊天历史
                gemini_history = []
                user_msgs = []
                
                for msg in context:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    
                    if role == "system":
                        # Gemini不支持系统消息，将其添加为用户消息
                        user_msgs.append(content)
                    elif role == "user":
                        user_msgs.append(content)
                    elif role == "assistant" and user_msgs:
                        # 添加用户消息和助手回复作为一对
                        gemini_history.append({
                            "role": "user", 
                            "parts": ["\n".join(user_msgs)]
                        })
                        user_msgs = []
                        gemini_history.append({
                            "role": "model",
                            "parts": [content]
                        })
                
                # 添加最后的用户消息
                if user_msgs or prompt:
                    final_user_msg = "\n".join(user_msgs + ([prompt] if prompt else []))
                    if final_user_msg:
                        if gemini_history:
                            gemini_history.append({"role": "user", "parts": [final_user_msg]})
                        else:
                            # 如果没有历史，直接使用提示
                            final_prompt = final_user_msg
                
                # 使用Gemini Chat API
                if gemini_history:
                    # 使用聊天历史
                    chat = await asyncio.to_thread(
                        lambda: self.genai.GenerativeModel(model).start_chat(history=gemini_history)
                    )
                    
                    response = await asyncio.to_thread(
                        lambda: chat.send_message(
                            "",  # 空消息，因为最后的用户消息已经在历史中
                            generation_config={
                                "temperature": temperature,
                                "max_output_tokens": max_tokens,
                            }
                        )
                    )
                else:
                    # 没有历史，直接使用提示
                    model_instance = await asyncio.to_thread(
                        lambda: self.genai.GenerativeModel(model)
                    )
                    
                    response = await asyncio.to_thread(
                        lambda: model_instance.generate_content(
                            prompt,
                            generation_config={
                                "temperature": temperature,
                                "max_output_tokens": max_tokens,
                            }
                        )
                    )
                
                # 提取文本内容
                return response.text
        except Exception as e:
            raise self._format_exception(e)
    
    async def generate_embeddings(self, texts: List[str], model: str = None) -> List[List[float]]:
        """生成文本嵌入向量
        
        Args:
            texts: 文本列表
            model: 嵌入模型名称，默认为embedding-001
            
        Returns:
            List[List[float]]: 嵌入向量列表
            
        Raises:
            LLMProviderException: Gemini API调用异常
        """
        try:
            if not model:
                model = "embedding-001"
            
            if self.use_langchain:
                # 使用LangChain集成
                from langchain_google_community import GoogleGenerativeAIEmbeddings
                
                embedding_model = GoogleGenerativeAIEmbeddings(
                    model=model,
                    google_api_key=self.api_key
                )
                
                # 使用LangChain嵌入
                embeddings = await asyncio.to_thread(embedding_model.embed_documents, texts)
                return embeddings
            else:
                # 使用Google GenerativeAI SDK直接处理
                # 处理单个文本嵌入
                async def get_embedding(text):
                    embedding_model = await asyncio.to_thread(
                        lambda: self.genai.GenerativeModel(model)
                    )
                    
                    result = await asyncio.to_thread(
                        lambda: embedding_model.embed_content(text)
                    )
                    
                    return result.embedding
                
                # 并行处理多个文本
                embeddings = await asyncio.gather(*[get_embedding(text) for text in texts])
                return embeddings
        except Exception as e:
            raise self._format_exception(e)
    
    async def count_tokens(self, text: str, model: str = None) -> Dict[str, int]:
        """计算文本的token数量
        
        Args:
            text: 待计算的文本
            model: 模型名称
            
        Returns:
            Dict[str, int]: 包含token数量和字符数量的字典
            
        Raises:
            LLMProviderException: 计算token异常
        """
        try:
            if not model:
                model = "gemini-pro"
            
            if self.use_langchain:
                # 使用LangChain集成计算token
                # 目前LangChain没有直接支持Google token计数，使用Google API
                if hasattr(self, 'genai'):
                    # 使用Google提供的CountTokens API
                    model_instance = await asyncio.to_thread(
                        lambda: self.genai.GenerativeModel(model)
                    )
                    
                    token_count = await asyncio.to_thread(
                        lambda: model_instance.count_tokens(text)
                    )
                    
                    return {
                        "token_count": token_count.total_tokens,
                        "character_count": len(text)
                    }
                else:
                    # 如果没有Google API，使用tiktoken作为后备
                    encoding = tiktoken.get_encoding("cl100k_base")
                    tokens = encoding.encode(text)
                    token_count = len(tokens)
                    
                    return {
                        "token_count": token_count,
                        "character_count": len(text)
                    }
            else:
                # 使用Google提供的CountTokens API
                model_instance = await asyncio.to_thread(
                    lambda: self.genai.GenerativeModel(model)
                )
                
                token_count = await asyncio.to_thread(
                    lambda: model_instance.count_tokens(text)
                )
                
                return {
                    "token_count": token_count.total_tokens,
                    "character_count": len(text)
                }
        except Exception as e:
            # 如果API调用失败，使用近似计算
            try:
                # 对于英文为主的文本，使用cl100k_base作为后备
                encoding = tiktoken.get_encoding("cl100k_base")
                tokens = encoding.encode(text)
                token_count = len(tokens)
                
                return {
                    "token_count": token_count,
                    "character_count": len(text)
                }
            except:
                # 最简单的估算方法
                return {
                    "token_count": len(text) // 4,  # 粗略估计每个token约4个字符
                    "character_count": len(text)
                }
    
    async def stream_chat(self, prompt: str, context: List[Dict], model: str,
                       temperature: float = 0.7, max_tokens: int = 1000) -> AsyncGenerator[str, None]:
        """流式生成聊天回复
        
        Args:
            prompt: 提示文本
            context: 上下文消息列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大生成令牌数
            
        Yields:
            str: 生成的文本片段
            
        Raises:
            LLMProviderException: Gemini API调用异常
        """
        try:
            if self.use_langchain:
                # 使用LangChain集成
                from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
                
                # 转换为LangChain消息
                messages = []
                for msg in context:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    
                    if role == "user":
                        messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        messages.append(AIMessage(content=content))
                    elif role == "system":
                        messages.append(SystemMessage(content=content))
                
                # 添加当前提示
                if prompt:
                    messages.append(HumanMessage(content=prompt))
                
                # 创建LangChain流式模型
                chat_model = self.chat_model_cls(
                    model=model,
                    google_api_key=self.api_key,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    convert_system_message_to_human=True,  # Gemini不直接支持系统消息
                    streaming=True
                )
                
                # 调用流式响应
                async for chunk in await asyncio.to_thread(chat_model.astream, messages):
                    if hasattr(chunk, "content"):
                        yield chunk.content
                    elif isinstance(chunk, dict) and "content" in chunk:
                        yield chunk["content"]
                    else:
                        yield str(chunk)
            else:
                # 使用Google GenerativeAI SDK
                
                # 转换为Gemini格式的聊天历史
                gemini_history = []
                user_msgs = []
                
                for msg in context:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    
                    if role == "system":
                        # Gemini不支持系统消息，将其添加为用户消息
                        user_msgs.append(content)
                    elif role == "user":
                        user_msgs.append(content)
                    elif role == "assistant" and user_msgs:
                        # 添加用户消息和助手回复作为一对
                        gemini_history.append({
                            "role": "user", 
                            "parts": ["\n".join(user_msgs)]
                        })
                        user_msgs = []
                        gemini_history.append({
                            "role": "model",
                            "parts": [content]
                        })
                
                # 添加最后的用户消息
                if user_msgs or prompt:
                    final_user_msg = "\n".join(user_msgs + ([prompt] if prompt else []))
                    if final_user_msg:
                        if gemini_history:
                            gemini_history.append({"role": "user", "parts": [final_user_msg]})
                        else:
                            # 如果没有历史，直接使用提示
                            final_prompt = final_user_msg
                
                # 使用异步生成器包装流式响应
                async def stream_response():
                    try:
                        # 使用Gemini Chat API
                        if gemini_history:
                            # 使用聊天历史
                            chat = await asyncio.to_thread(
                                lambda: self.genai.GenerativeModel(model).start_chat(history=gemini_history)
                            )
                            
                            response_iter = await asyncio.to_thread(
                                lambda: chat.send_message(
                                    "",  # 空消息，因为最后的用户消息已经在历史中
                                    generation_config={
                                        "temperature": temperature,
                                        "max_output_tokens": max_tokens,
                                    },
                                    stream=True
                                )
                            )
                        else:
                            # 没有历史，直接使用提示
                            model_instance = await asyncio.to_thread(
                                lambda: self.genai.GenerativeModel(model)
                            )
                            
                            response_iter = await asyncio.to_thread(
                                lambda: model_instance.generate_content(
                                    prompt,
                                    generation_config={
                                        "temperature": temperature,
                                        "max_output_tokens": max_tokens,
                                    },
                                    stream=True
                                )
                            )
                        
                        # 处理流式响应
                        for chunk in response_iter:
                            if chunk.text:
                                yield chunk.text
                    except Exception as e:
                        raise self._format_exception(e)
                
                # 返回异步生成器
                async for chunk in stream_response():
                    yield chunk
        except Exception as e:
            raise self._format_exception(e)
            
    def _get_model_defaults(self, model: str, task_type: str = "chat") -> Dict[str, Any]:
        """获取模型默认参数
        
        Args:
            model: 模型名称
            task_type: 任务类型
            
        Returns:
            Dict[str, Any]: 模型默认参数
        """
        if task_type == "embedding":
            return {
                "max_tokens": 768,  # Gemini嵌入向量大小
            }
        
        # Gemini模型默认参数
        if "gemini-ultra" in model:
            return {
                "temperature": 0.7,
                "max_tokens": 4000,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }
        elif "gemini-pro" in model:
            return {
                "temperature": 0.7,
                "max_tokens": 2048,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }
        elif "gemini-" in model:
            return {
                "temperature": 0.7,
                "max_tokens": 1024,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }
        else:
            return super()._get_model_defaults(model, task_type)
