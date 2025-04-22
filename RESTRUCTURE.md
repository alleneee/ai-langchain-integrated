# Dify-Connect 项目重构说明

## 项目结构重构

本项目已根据FastAPI最佳实践进行了重构，主要改进包括：

1. **目录结构优化**：采用更合理的分层架构
2. **依赖注入模式**：使用FastAPI依赖注入进行服务管理
3. **错误处理统一化**：添加全局异常处理机制
4. **规范化API响应**：统一响应格式
5. **代码组织优化**：更清晰的模块划分

## 新的目录结构

```
src/
├── api/                    # API路由和接口定义
│   ├── endpoints/          # 具体API端点实现
│   └── router.py           # 路由器统一配置
├── config/                 # 配置管理
│   └── settings.py         # 应用配置
├── core/                   # 核心功能
│   └── exceptions.py       # 自定义异常
├── db/                     # 数据库相关
├── dependencies/           # 依赖项
│   └── common.py           # 通用依赖函数
├── middlewares/            # 中间件
│   └── error_handler.py    # 错误处理中间件
├── models/                 # 数据模型
├── repositories/           # 数据仓库
├── schemas/                # 请求和响应模式
│   ├── base.py             # 基础模式
│   ├── chat.py             # 聊天相关模式
│   ├── responses.py        # 响应模式
│   └── ...                 # 其他模式
├── services/               # 服务层
│   ├── base.py             # 基础服务类
│   ├── chat_service.py     # 聊天服务
│   └── ...                 # 其他服务
├── utils/                  # 工具函数
│   └── token_counter.py    # 令牌计数工具
├── __init__.py             # 包初始化
└── main.py                 # 应用入口
```

## 主要改进

### 1. 依赖注入

使用FastAPI的Depends系统进行依赖管理，使服务和配置更容易测试和替换。

```python
@router.get("/providers")
async def get_supported_providers(settings = Depends(get_settings_dependency)):
    # 使用注入的设置
```

### 2. 统一异常处理

添加了全局异常处理中间件，所有API错误都能以统一格式返回给客户端。

```python
app.add_exception_handler(AppException, app_exception_handler)
```

### 3. 规范化API响应

创建标准响应模式，所有API都使用统一的响应格式。

```python
{
    "success": true,
    "message": "操作成功",
    "data": { ... }
}
```

### 4. 类型安全

使用Pydantic和类型注解确保代码的类型安全性。

## 后续工作

1. 实现各个服务的具体功能
2. 添加数据库模型和迁移
3. 添加认证和权限控制
4. 添加更多测试
5. 优化性能

## 参考资料

- [FastAPI官方文档](https://fastapi.tiangolo.com/)
- [FastAPI最佳实践](https://github.com/zhanymkanov/fastapi-best-practices)
