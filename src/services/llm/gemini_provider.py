"""
Google Gemini LLM提供商适配器

实现了Google Gemini API的调用与适配，使用langchain-google-community库
"""

import json
import tiktoken
import asyncio
from typing import Dict, List, Any, Optional, AsyncGenerator

from src.services.llm.base import BaseLLMProvider
from src.utils.llm_exception_utils import handle_llm_exception
from src.utils.async_utils import run_sync_in_executor

# Potential imports - will be used conditionally
try:
    from langchain_google_community import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    import google.generativeai as genai
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False

class GeminiProvider(BaseLLMProvider):
    """Google Gemini API适配器"""
    
    def __init__(self, api_key: str = None, api_base: str = None, **kwargs):
        """初始化Gemini适配器"""
        super().__init__(api_key, api_base, **kwargs)
        self.genai = None
        self.chat_model_cls = None
        self.embedding_model_cls = None
        self.use_langchain = False
        self.client_ready = False
        try:
            self._setup_client()
            self.client_ready = True
        except Exception as e:
            handle_llm_exception(e, self.provider_name)
            # Keep client_ready as False

    def _setup_client(self):
        """设置Gemini客户端 (Langchain 或 SDK)"""
        self._validate_api_key() # Will raise if key is missing
            
        if LANGCHAIN_AVAILABLE:
            self.chat_model_cls = ChatGoogleGenerativeAI
            self.embedding_model_cls = GoogleGenerativeAIEmbeddings
            self.use_langchain = True
            print("GeminiProvider: Using Langchain integration.")
            # Langchain lazy loads the connection, setup is minimal here.
            # Validation happens on first call.
        elif SDK_AVAILABLE:
            print("GeminiProvider: Langchain-google-community not found, falling back to google-generativeai SDK.")
            # Configure the SDK globally (as recommended by google-generativeai)
            try:
                genai.configure(api_key=self.api_key)
                self.genai = genai 
                self.use_langchain = False
                # Test connection (optional but recommended)
                # list_models call is relatively lightweight
                list(self.genai.list_models())
            except Exception as sdk_e:
                 # Catch configuration or initial connection errors
                 raise RuntimeError(f"Failed to configure or connect using google-generativeai SDK: {sdk_e}") from sdk_e
        else:
            raise ImportError("Neither langchain-google-community nor google-generativeai library is installed. Please install one.")

    def _ensure_client_ready(self):
        if not self.client_ready:
             raise RuntimeError(f"{self.provider_name} client failed to initialize. Check configuration, API key, and installed libraries.")

    async def generate_text(self, prompt: str, context: List[Dict], model: str,
                         temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """生成文本响应"""
        self._ensure_client_ready()
        try:
            if self.use_langchain and self.chat_model_cls:
                # --- LangChain Implementation ---
                from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
                
                messages = []
                for msg in context:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    # Gemini Langchain converts System message if needed
                    if role == "user": messages.append(HumanMessage(content=content))
                    elif role == "assistant": messages.append(AIMessage(content=content))
                    elif role == "system": messages.append(SystemMessage(content=content))
                if prompt: messages.append(HumanMessage(content=prompt))
                
                chat_model = self.chat_model_cls(
                    model=model,
                    google_api_key=self.api_key,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    convert_system_message_to_human=True 
                )
                
                # Use run_sync_in_executor for the synchronous invoke call
                response = await run_sync_in_executor(chat_model.invoke, messages)
                
                if hasattr(response, "content"): return response.content
                else: return str(response)

            elif not self.use_langchain and self.genai:
                # --- Google GenerativeAI SDK Implementation ---
                gemini_history = []
                current_user_message_parts = []

                # Process context into Gemini format (alternating user/model)
                last_role = None
                for msg in context:
                    role = msg.get("role")
                    content = msg.get("content", "")
                    if not role or not content:
                        continue
                    
                    # Combine consecutive user messages
                    if role == "user":
                        if last_role == "user":
                            current_user_message_parts.append(content)
                        else:
                            current_user_message_parts = [content]
                    elif role == "assistant": # 'model' in gemini terms
                        # Finalize previous user message block if it exists
                        if current_user_message_parts:
                            gemini_history.append({"role": "user", "parts": ["\n".join(current_user_message_parts)]})
                            current_user_message_parts = []
                        gemini_history.append({"role": "model", "parts": [content]})
                    
                    last_role = role

                # Add the final prompt to the last user message block or as a new one
                if prompt:
                     if last_role == "user":
                         current_user_message_parts.append(prompt)
                     else:
                         current_user_message_parts = [prompt]
                
                # Prepare generation config
                generation_config = self.genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
                
                # Get the model instance (synchronous)
                model_instance = self.genai.GenerativeModel(model)
                
                # Call generate_content (synchronous)
                if gemini_history:
                    # Start chat and send message
                    chat = await run_sync_in_executor(model_instance.start_chat, history=gemini_history)
                    response = await run_sync_in_executor(
                        chat.send_message,
                        final_user_msg,
                        generation_config=generation_config
                    )
                else:
                     # No history, just generate from prompt
                    response = await run_sync_in_executor(
                        model_instance.generate_content, 
                        final_user_msg, 
                        generation_config=generation_config
                    )
                
                # Extract text
                if response and hasattr(response, 'text'):
                    return response.text
                else:
                    # Handle cases where response might be blocked or empty
                    # Check for safety ratings or prompt feedback if needed
                    if response and hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                         raise RuntimeError(f"Gemini API request blocked due to: {response.prompt_feedback.block_reason}")
                    raise RuntimeError("Gemini API response format unexpected or empty.")
            else:
                 raise RuntimeError(f"{self.provider_name} client not properly configured or libraries missing.")
        except Exception as e:
            handle_llm_exception(e, self.provider_name)
    
    async def generate_embeddings(self, texts: List[str], model: str = None) -> List[List[float]]:
        """生成文本嵌入向量"""
        self._ensure_client_ready()
        embedding_model_name = model if model else "models/embedding-001" # Default Gemini embedding model
        try:
            if self.use_langchain and self.embedding_model_cls:
                 # --- LangChain Implementation ---
                embedding_model = self.embedding_model_cls(
                    model_name=embedding_model_name, # Langchain uses model_name
                    google_api_key=self.api_key
                    # task_type can be specified if needed, default is RETRIEVAL_DOCUMENT
                )
                # Use run_sync_in_executor for the synchronous embed_documents call
                embeddings = await run_sync_in_executor(embedding_model.embed_documents, texts)
                return embeddings
            
            elif not self.use_langchain and self.genai:
                 # --- Google GenerativeAI SDK Implementation ---
                 # SDK's embed_content handles batching implicitly up to limits
                 # task_type: 'RETRIEVAL_QUERY', 'RETRIEVAL_DOCUMENT', 'SEMANTIC_SIMILARITY', 'CLASSIFICATION', 'CLUSTERING'
                 # Assuming 'RETRIEVAL_DOCUMENT' as a general default
                 response = await run_sync_in_executor(
                     self.genai.embed_content, 
                     model=embedding_model_name,
                     content=texts,
                     task_type="RETRIEVAL_DOCUMENT"
                 )
                 
                 if response and 'embedding' in response:
                     # If single text, response['embedding'] is the list
                     # If multiple texts, response['embedding'] is a list of lists -> WRONG, SDK returns list of lists
                     # Let's check structure carefully based on SDK docs (it should return list of lists for multiple inputs)
                      if isinstance(response['embedding'], list) and len(response['embedding']) > 0: 
                          if isinstance(response['embedding'][0], list):
                                # Multiple embeddings returned as list of lists
                                return response['embedding']
                          else:
                                # Single embedding returned as list
                                return [response['embedding']] # Wrap in a list
                      else:
                          raise RuntimeError("Gemini API embedding response format unexpected or empty.")
                 else:
                     raise RuntimeError("Gemini API embedding response format unexpected or missing 'embedding' key.")
            else:
                 raise RuntimeError(f"{self.provider_name} client not properly configured or libraries missing.")
        except Exception as e:
            handle_llm_exception(e, self.provider_name)

    async def count_tokens(self, text: str, model: str = None) -> Dict[str, int]:
        """计算文本的token数量"""
        # Note: Gemini token counting is best done via the API for accuracy.
        # Tiktoken is not officially supported and likely inaccurate for Gemini models.
        self._ensure_client_ready()
        target_model = model if model else "gemini-pro" # Need a model context
        try:
            if self.use_langchain and self.chat_model_cls: 
                 # Langchain Chat model instance can count tokens
                 # Need to instantiate the model to call count_tokens
                 # This is slightly inefficient if not already generating text
                 chat_model = self.chat_model_cls(model=target_model, google_api_key=self.api_key)
                 # Langchain's get_num_tokens is synchronous
                 token_count = await run_sync_in_executor(chat_model.get_num_tokens, text)
                 return {"token_count": token_count, "char_count": len(text)}
            
            elif not self.use_langchain and self.genai:
                # --- Google GenerativeAI SDK Implementation ---
                model_instance = self.genai.GenerativeModel(target_model)
                # count_tokens is synchronous
                response = await run_sync_in_executor(model_instance.count_tokens, text)
                
                if response and hasattr(response, 'total_tokens'):
                    return {"token_count": response.total_tokens, "char_count": len(text)}
                else:
                    raise RuntimeError("Gemini API count_tokens response format unexpected.")
            else:
                 raise RuntimeError(f"{self.provider_name} client not properly configured or libraries missing.")

        except Exception as e:
            # Handle API errors or if counting fails for the model
            handle_llm_exception(e, f"{self.provider_name} (token counting)")

    async def stream_chat(self, prompt: str, context: List[Dict], model: str,
                       temperature: float = 0.7, max_tokens: int = 1000) -> AsyncGenerator[str, None]:
        """流式生成聊天回复"""
        self._ensure_client_ready()
        try:
            if self.use_langchain and self.chat_model_cls:
                # --- LangChain Implementation ---
                from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
                
                messages = []
                for msg in context:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user": messages.append(HumanMessage(content=content))
                    elif role == "assistant": messages.append(AIMessage(content=content))
                    elif role == "system": messages.append(SystemMessage(content=content))
                if prompt: messages.append(HumanMessage(content=prompt))
                
                chat_model = self.chat_model_cls(
                    model=model,
                    google_api_key=self.api_key,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    convert_system_message_to_human=True,
                    streaming=True # Ensure streaming is enabled
                )
                
                # Langchain's astream should yield an async generator
                async for chunk in chat_model.astream(messages):
                    if hasattr(chunk, "content"): yield chunk.content
                    elif isinstance(chunk, dict) and "content" in chunk: yield chunk["content"]
                    elif isinstance(chunk, str): yield chunk
                    # Ignore other potential chunk types

            elif not self.use_langchain and self.genai:
                # --- Google GenerativeAI SDK Implementation ---
                gemini_history = []
                current_user_message_parts = []
                last_role = None
                for msg in context:
                    role = msg.get("role")
                    content = msg.get("content", "")
                    if not role or not content: continue
                    if role == "user":
                        if last_role == "user": current_user_message_parts.append(content)
                        else: current_user_message_parts = [content]
                    elif role == "assistant":
                        if current_user_message_parts: 
                            gemini_history.append({"role": "user", "parts": ["\n".join(current_user_message_parts)]})
                            current_user_message_parts = []
                        gemini_history.append({"role": "model", "parts": [content]})
                    last_role = role
                if prompt:
                     if last_role == "user": current_user_message_parts.append(prompt)
                     else: current_user_message_parts = [prompt]
                final_user_msg = ""
                if current_user_message_parts:
                    final_user_msg = "\n".join(current_user_message_parts)

                generation_config = self.genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
                model_instance = self.genai.GenerativeModel(model)
                
                # Get the synchronous stream iterator using run_sync_in_executor
                if gemini_history:
                    chat = await run_sync_in_executor(model_instance.start_chat, history=gemini_history)
                    response_iter = await run_sync_in_executor(
                        chat.send_message,
                        final_user_msg,
                        generation_config=generation_config,
                        stream=True
                    )
                else:
                     response_iter = await run_sync_in_executor(
                         model_instance.generate_content, 
                         final_user_msg, 
                         generation_config=generation_config,
                         stream=True
                     )
                
                # Asynchronously iterate over the synchronous iterator
                iter_ended = False
                while not iter_ended:
                    try:
                        # Get next chunk in executor
                        chunk = await run_sync_in_executor(next, response_iter)
                        if chunk and hasattr(chunk, 'text') and chunk.text:
                            yield chunk.text
                    except StopIteration:
                        iter_ended = True
                    except Exception as inner_e:
                        # Handle errors during iteration
                        handle_llm_exception(inner_e, self.provider_name)
                        iter_ended = True # Stop iteration on error
            else:
                raise RuntimeError(f"{self.provider_name} client not properly configured or libraries missing.")

        except Exception as e:
            handle_llm_exception(e, self.provider_name)
            
    def _get_model_defaults(self, model: str, task_type: str = "chat") -> Dict[str, Any]:
        """获取模型默认参数 (保持不变)"""
        if task_type == "embedding":
            # Default embedding size (output dimension) - max_tokens isn't applicable here
            # Input max tokens depend on the model, e.g., 2048 for embedding-001
            return {
                 # "max_input_tokens": 2048, # Informational
            }
        
        defaults = {
            "temperature": 0.7,
            "top_p": 1.0,
            "frequency_penalty": 0.0, # Not directly supported by Gemini API
            "presence_penalty": 0.0   # Not directly supported by Gemini API
        }
        
        # Max output tokens defaults based on common Gemini models
        if "gemini-1.5" in model or "gemini-ultra" in model: # Ultra / 1.5 Flash/Pro
            defaults["max_tokens"] = 8192 # Check specific model docs, 1.5 can be much higher
        elif "gemini-1.0-pro" in model or "gemini-pro" in model: # 1.0 Pro
            defaults["max_tokens"] = 2048 # Or 8192 for gemini-1.0-pro-002
        else:
             # Generic fallback
             defaults["max_tokens"] = 1024
        
        # Gemini API might have different param names or concepts (e.g., top_k)
        # Adjust if necessary based on specific API usage
        # Example: adding top_k if used directly via SDK
        # defaults["top_k"] = 40 

        return defaults
