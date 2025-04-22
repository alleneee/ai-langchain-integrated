"""
Celery 应用模块

创建和配置 Celery 应用实例
"""

import os
import sys
from celery import Celery

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# 创建 Celery 应用
app = Celery('dify_connect')

# 从对象加载配置
app.config_from_object('src.celery_app.celery_config')

# 自动发现任务
app.autodiscover_tasks([
    'src.celery_app.tasks',
])

# 启动时执行的操作
@app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    """设置定期任务"""
    # 可以在这里添加定期任务
    # 例如：清理过期的文档处理结果
    # sender.add_periodic_task(
    #     60 * 60 * 24,  # 24小时
    #     clean_expired_results.s(),
    #     name='clean expired results every 24 hours'
    # )
    pass

if __name__ == '__main__':
    app.start()
