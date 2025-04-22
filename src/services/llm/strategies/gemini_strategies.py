"""
LLM Interaction Strategy for Google Gemini Provider.
"""

import os
from typing import List, Dict, Any, Optional, AsyncGenerator

# Gemini uses specific Langchain integrations
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage # Use Langchain core messages
# Note: Gemini might have nuances with SystemMessage roles.

from src.services.llm.base import LLMStrategy
from src.utils.llm_exception_utils import handle_llm_exception

# Gemini uses GOOGLE_API_KEY
_GEMINI_API_KEY_NAME = "GOOGLE_API_KEY"
_DEFAULT_GEMINI_CHAT_MODEL = "gemini-1.5-flash-latest" # Use a known default
_DEFAULT_GEMINI_EMBEDDING_MODEL = "models/embedding-001" # Use the recommended embedding model

class GeminiLangchainStrategy(LLMStrategy):
    """LLM Strategy implementation using Langchain for Google Gemini."""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        self.api_key = api_key or os.getenv(_GEMINI_API_KEY_NAME)
        self.extra_kwargs = kwargs

        if not self.api_key:
            raise ValueError(f"{_GEMINI_API_KEY_NAME} is required for GeminiLangchainStrategy.")

        # Gemini token counting is often handled by the client library or needs specific logic.
        # We won't use tiktoken here as it's likely inaccurate for Gemini.

    def _create_chat_client(self, model: str, temperature: float, max_tokens: int) -> ChatGoogleGenerativeAI:
        # Gemini uses 'max_output_tokens' instead of 'max_tokens'
        return ChatGoogleGenerativeAI(
            model=model or _DEFAULT_GEMINI_CHAT_MODEL,
            google_api_key=self.api_key,
            temperature=temperature,
            max_output_tokens=max_tokens, # Use the correct parameter name
            convert_system_message_to_human=True, # Often needed for Gemini compatibility
            **(self.extra_kwargs.get("chat_options", {}))
        )

    def _create_embedding_client(self, model: Optional[str] = None) -> GoogleGenerativeAIEmbeddings:
        embedding_model_name = model or _DEFAULT_GEMINI_EMBEDDING_MODEL
        return GoogleGenerativeAIEmbeddings(
            model=embedding_model_name,
            google_api_key=self.api_key,
            **(self.extra_kwargs.get("embedding_options", {}))
        )

    def _prepare_langchain_messages(self, context: List[Dict], prompt: str) -> List:
        # Adapt message prep if Gemini has specific requirements (e.g., system message handling)
        messages = []
        system_content = ""
        for msg in context:
            role = msg.get("role", "user").lower()
            content = msg.get("content", "")
            if content:
                if role == "system":
                    # Gemini often prefers system instructions prepended to the first human message
                    # or uses specific turn structures. Using SystemMessage might require
                    # 'convert_system_message_to_human=True' in the client.
                    system_content += content + "\n"
                    messages.append(SystemMessage(content=content))
                elif role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant" or role == "ai":
                    messages.append(AIMessage(content=content))
        
        # Prepend system content to prompt if necessary, or rely on SystemMessage + client conversion
        final_prompt_content = prompt # Modify if system content needs merging
        # if system_content and not any(isinstance(m, SystemMessage) for m in messages):
             # Example: Prepend if no SystemMessage was added and conversion is off
             # final_prompt_content = system_content + prompt

        if final_prompt_content:
             messages.append(HumanMessage(content=final_prompt_content))
             
        # Simplified: Relying on SystemMessage and convert_system_message_to_human=True
        # Ensure the order is reasonable (e.g., System, Human, AI, Human...) 
        # Langchain client usually handles this.
        return messages

    @handle_llm_exception
    async def generate_text_async(self, prompt: str, context: List[Dict], model: str, temperature: float, max_tokens: int, **kwargs) -> str:
        llm = self._create_chat_client(model, temperature, max_tokens)
        messages = self._prepare_langchain_messages(context, prompt)
        response = await llm.ainvoke(messages, **kwargs)
        return response.content

    @handle_llm_exception
    async def generate_embeddings_async(self, texts: List[str], model: Optional[str] = None, **kwargs) -> List[List[float]]:
        embedder = self._create_embedding_client(model)
        # Gemini embeddings might have specific batching or input requirements
        return await embedder.aembed_documents(texts, **kwargs)

    @handle_llm_exception
    async def count_tokens_async(self, text: str, model: Optional[str] = None, **kwargs) -> Dict[str, int]:
        # Use the client's method for token counting if available and accurate
        client_for_counting = ChatGoogleGenerativeAI(model=model or _DEFAULT_GEMINI_CHAT_MODEL, google_api_key=self.api_key)
        try:
            # Note: get_num_tokens might operate on messages, not raw text. Adapt as needed.
            # This is an approximation using a single HumanMessage.
            token_count = client_for_counting.get_num_tokens([HumanMessage(content=text)])
        except Exception as e:
            print(f"Warning: Failed to get token count from Gemini client: {e}. Returning 0.")
            token_count = 0 # Fallback or raise
        return {"token_count": token_count, "character_count": len(text)}

    @handle_llm_exception
    async def stream_chat_async(self, prompt: str, context: List[Dict], model: str, temperature: float, max_tokens: int, **kwargs) -> AsyncGenerator[str, None]:
        llm = self._create_chat_client(model, temperature, max_tokens)
        messages = self._prepare_langchain_messages(context, prompt)
        async for chunk in llm.astream(messages, **kwargs):
            if chunk.content:
                 yield chunk.content

    def get_models(self) -> List[str]:
        # Provide known Gemini models
        return [
            "gemini-1.5-pro-latest", "gemini-1.5-flash-latest",
            "gemini-1.0-pro", "gemini-pro", # Alias
            "models/embedding-001", # Correct embedding model name
            "models/aqa" # Example specialized model
        ]
