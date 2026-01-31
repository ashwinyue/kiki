"""核心模块

提供配置、错误处理、认证、限流、中间件等核心功能。

模块说明：
- config: 配置管理
- configuration: 运行时配置
- errors: 错误处理
- auth: 认证相关
- limiter: 限流
- middleware: 中间件
- memory: 上下文存储
- search: 搜索服务
- store: 状态存储
- evaluation: 评估模块
- dependencies: 依赖注入
"""

# 注意：不导入 dependencies 避免循环依赖
# dependencies 需要在运行时动态导入

__all__ = [
    "config",
    "configuration",
    "errors",
    "auth",
    "limiter",
    "middleware",
    "memory",
    "search",
    "store",
    "evaluation",
    "dependencies",
]
