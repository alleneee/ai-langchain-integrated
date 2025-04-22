"""
令牌计数工具模块

该模块提供了估计文本令牌数的工具函数。
"""

from typing import Dict, Any

def estimate_token_count(text: str) -> int:
    """估计文本的令牌数
    
    通常GPT模型中，1个token约等于4个字符或0.75个单词
    对于中文，一个汉字约等于一个token
    
    Args:
        text: 要估计的文本
        
    Returns:
        估计的令牌数
    """
    # 简单版本：对于英文，大约4个字符一个token；对于中文，约1个字符1个token
    chinese_char_count = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    other_char_count = len(text) - chinese_char_count
    
    # 估算token数量
    token_count = chinese_char_count + other_char_count // 4
    
    return token_count 