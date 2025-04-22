"""
基础服务模块

该模块定义了所有服务的基类。
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel
import requests
from requests import Response
from abc import ABC, abstractmethod

class BaseService(ABC):
    """服务基类
    
    所有服务类都应该继承此基类
    """
    
    @abstractmethod
    async def initialize(self):
        """初始化服务"""
        pass
    
    def _process_response(self, response: Response) -> Dict[str, Any]:
        """处理HTTP响应
        
        Args:
            response: HTTP响应对象
            
        Returns:
            处理后的响应数据
        """
        if response.status_code >= 400:
            return {
                "success": False,
                "message": f"请求失败: {response.status_code}",
                "data": response.json() if self._is_json(response) else {"raw": response.text}
            }
        
        return {
            "success": True,
            "message": "请求成功",
            "data": response.json() if self._is_json(response) else {"raw": response.text}
        }
    
    def _is_json(self, response: Response) -> bool:
        """检查响应是否为JSON格式
        
        Args:
            response: HTTP响应对象
            
        Returns:
            是否为JSON格式
        """
        try:
            response.json()
            return True
        except:
            return False 