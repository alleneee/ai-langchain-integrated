"""
通义千问(Qwen) LLM提供商适配器

实现了阿里云通义千问 API的调用与适配
"""

import json
import tiktoken
import asyncio
from typing import Dict, List, Any, Optional, AsyncGenerator

from src.services.llm.base import BaseLLMProvider
from src.utils.llm_exception_utils import handle_llm_exception
from src.utils.async_utils import run_sync_in_executor

# Dashscope SDK
import dashscope
from dashscope import Generation, TextEmbedding

class QwenProvider(BaseLLMProvider):
    """通义千问 API适配器"""
    
    def __init__(self, api_key: str = None, api_base: str = None, **kwargs):
        """初始化通义千问适配器"""
        super().__init__(api_key, api_base, **kwargs)
        # Dashscope SDK 使用全局 API key，在 setup 中设置
        try:
            self._setup_client()
            self.client_ready = True
        except Exception as e:
            handle_llm_exception(e, self.provider_name)
            self.client_ready = False

    def _setup_client(self):
        """设置通义千问客户端 (实际是设置全局 API Key)"""
        self._validate_api_key() # 验证 key 是否存在
        # 设置DashScope API密钥
        dashscope.api_key = self.api_key
        # 可以在这里添加一个简单的测试调用来验证 API Key 的有效性 (可选)
        # try:
        #     # Example: Make a simple call to test connectivity/auth
        #     Generation.call(model='qwen-turbo', prompt='Test Connection', max_tokens=1)
        # except Exception as setup_e:
        #     raise LLMAuthenticationError(f"Failed to validate Qwen API key: {setup_e}") from setup_e

    def _ensure_client_ready(self):
        if not self.client_ready:
            raise RuntimeError(f"{self.provider_name} client failed to initialize or API key is invalid. Check configuration and logs.")

    async def generate_text(self, prompt: str, context: List[Dict], model: str,
                         temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """生成文本响应"""
        self._ensure_client_ready()
        try:
            messages = self._prepare_messages(context, prompt)
            dashscope_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
            
            # 定义要传递给 Generation.call 的参数
            call_args = {                
                "model": model,
                "messages": dashscope_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "result_format": 'message', # 确保返回 message 结构
            }
            
            # 使用 run_sync_in_executor 运行同步 DashScope API
            response = await run_sync_in_executor(Generation.call, **call_args)
            
            # 检查响应
            if response.status_code == 200:
                if response.output and response.output.choices and len(response.output.choices) > 0:
                    return response.output.choices[0].message.content
                else:
                    raise RuntimeError("Qwen API response format unexpected (no output or choices)")
            else:
                # Dashscope SDK 应该在失败时抛出异常，但以防万一
                # 创建一个模拟的异常对象，传递信息给 handle_llm_exception
                simulated_exc = Exception(f"Qwen API Error: Code {response.code}, Message: {response.message}")
                setattr(simulated_exc, 'status_code', response.status_code) 
                setattr(simulated_exc, 'response', response) # 附加完整响应供调试
                raise simulated_exc

        except Exception as e:
            handle_llm_exception(e, self.provider_name)
    
    async def generate_embeddings(self, texts: List[str], model: str = None) -> List[List[float]]:
        """生成文本嵌入向量"""
        self._ensure_client_ready()
        embedding_model_name = model if model else "text-embedding-v1"
        try:
            # 定义要传递给 TextEmbedding.call 的参数
            call_args = {
                "model": embedding_model_name,
                "input": texts,
                "text_type": "document" # 或 "query"，取决于用途
            }

            # 使用 run_sync_in_executor 运行同步 DashScope API
            response = await run_sync_in_executor(TextEmbedding.call, **call_args)
            
            if response.status_code == 200:
                if response.output and response.output.embeddings:
                    return [item.embedding for item in response.output.embeddings]
                else:
                    raise RuntimeError("Qwen API embedding response format unexpected (no output or embeddings)")
            else:
                simulated_exc = Exception(f"Qwen Embedding API Error: Code {response.code}, Message: {response.message}")
                setattr(simulated_exc, 'status_code', response.status_code)
                setattr(simulated_exc, 'response', response)
                raise simulated_exc

        except Exception as e:
            handle_llm_exception(e, self.provider_name)

    async def count_tokens(self, text: str, model: str = None) -> Dict[str, int]:
        """计算文本的token数量"""
        # Qwen 使用 cl100k_base 编码器
        try:
            # 使用 run_sync_in_executor 包装同步调用
            encoding = await run_sync_in_executor(tiktoken.get_encoding, "cl100k_base")
            tokens = await run_sync_in_executor(encoding.encode, text)
            token_count = len(tokens)
            char_count = len(text)
            return {"token_count": token_count, "char_count": char_count}
        except Exception as e:
             # tiktoken 本身可能出错
            handle_llm_exception(e, f"{self.provider_name} (tiktoken)")

    async def stream_chat(self, prompt: str, context: List[Dict], model: str,
                       temperature: float = 0.7, max_tokens: int = 1000) -> AsyncGenerator[str, None]:
        """流式生成聊天回复"""
        self._ensure_client_ready()
        try:
            messages = self._prepare_messages(context, prompt)
            dashscope_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]

            call_args = {
                "model": model,
                "messages": dashscope_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "result_format": 'message',
                "stream": True,
                "incremental_output": True # 获取增量输出
            }

            # 使用 run_sync_in_executor 来获取同步迭代器
            response_iter = await run_sync_in_executor(Generation.call, **call_args)
            
            # 异步迭代同步迭代器
            # 这个模式有点奇怪，但对于包装同步生成器是必要的
            # 更好的方法是库本身提供异步接口
            iter_ended = False
            while not iter_ended:
                try:
                    # 在执行器中运行 next(iterator)
                    event = await run_sync_in_executor(next, response_iter)
                    if event.status_code == 200:
                        if event.output and event.output.choices and len(event.output.choices) > 0:
                            content = event.output.choices[0].message.content
                            if content: # 只 yield 非空内容
                                yield content
                    else:
                        # 如果流内部出错，也模拟异常抛出
                        simulated_exc = Exception(f"Qwen Streaming API Error: Code {event.code}, Message: {event.message}")
                        setattr(simulated_exc, 'status_code', event.status_code)
                        setattr(simulated_exc, 'response', event)
                        raise simulated_exc
                except StopIteration:
                    iter_ended = True # 迭代器正常结束
                except Exception as inner_e: 
                    # 捕获 next() 或流处理中的错误，并使用主处理器
                    # 避免下面再次捕获
                    handle_llm_exception(inner_e, self.provider_name)
                    iter_ended = True # 出错则停止迭代
                    
        except Exception as e:
            # 捕获 setup 或 run_sync_in_executor(Generation.call) 本身的错误
            handle_llm_exception(e, self.provider_name)
            
    def _get_model_defaults(self, model: str, task_type: str = "chat") -> Dict[str, Any]:
        """获取模型默认参数 (保持不变)"""
        if task_type == "embedding":
            return {
                "max_tokens": 2048, # text-embedding-v1 max length
            }
        
        # 通义千问模型默认参数 (根据官方文档或最佳实践调整)
        defaults = {
            "temperature": 0.85, # Qwen 推荐
            "top_p": 0.8,      # Qwen 推荐
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
        if "qwen-turbo" in model:
            defaults["max_tokens"] = 6000 # 根据模型调整
        elif "qwen-plus" in model:
            defaults["max_tokens"] = 6000
        elif "qwen-max" in model or "qwen-long" in model:
             defaults["max_tokens"] = 6000 # 检查具体模型限制
        else:
            return super()._get_model_defaults(model, task_type)
        
        return defaults
