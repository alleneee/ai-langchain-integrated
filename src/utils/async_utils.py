import asyncio
import functools
from typing import Any, Callable, Coroutine, TypeVar

ResultT = TypeVar('ResultT')

def run_sync_in_executor(func: Callable[..., ResultT], *args: Any, **kwargs: Any) -> Coroutine[Any, Any, ResultT]:
    """通用函数，用于在线程池执行器中运行同步函数。"""
    loop = asyncio.get_running_loop()
    return loop.run_in_executor(
        None,  # 使用默认执行器
        functools.partial(func, *args, **kwargs)
    )

# 你可以在这里添加其他异步工具函数
