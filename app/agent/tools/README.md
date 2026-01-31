# 工具开发指南

本目录包含 Agent 工具系统的实现和使用指南。

## 目录结构

```
tools/
├── __init__.py       # 统一导出，自动注册内置工具
├── registry.py       # 工具注册系统
├── README.md         # 本文件
└── builtin/          # 内置示例工具
    ├── __init__.py
    ├── database.py   # 数据库搜索工具
    ├── weather.py    # 天气查询工具
    └── calculation.py # 数学计算工具
```

## 创建自定义工具

### 方式 1: 使用装饰器（推荐）

```python
# app/core/agent/tools/custom/my_tool.py
from app.core.agent.tools import tool
from app.core.logging import get_logger

logger = get_logger(__name__)

@tool
async def my_custom_tool(query: str) -> str:
    """工具描述（LLM 可见）

    Args:
        query: 查询参数

    Returns:
        结果描述
    """
    logger.info("my_custom_tool_called", query=query)
    return f"处理结果: {query}"
```

### 方式 2: 手动注册

```python
from langchain_core.tools import tool
from app.core.agent.tools import register_tool

@tool
async def my_tool(query: str) -> str:
    """工具描述"""
    return f"结果: {query}"

# 手动注册
register_tool(my_tool)
```

## 工具最佳实践

### 1. 清晰的函数文档

工具的 docstring 会被 LLM 读取，确保描述清晰：

```python
@tool
async def search_user(user_id: str) -> str:
    """根据用户 ID 搜索用户信息

    Args:
        user_id: 用户唯一标识符

    Returns:
        JSON 格式的用户信息字符串
    """
    ...
```

### 2. 类型注解

使用类型注解帮助 LLM 理解参数类型：

```python
from typing import Literal

@tool
async def set_temperature(
    location: str,
    unit: Literal["celsius", "fahrenheit"] = "celsius",
) -> str:
    ...
```

### 3. 错误处理

妥善处理错误并返回友好的错误消息：

```python
@tool
async def query_database(sql: str) -> str:
    """执行数据库查询"""
    try:
        result = await db.execute(sql)
        return json.dumps(result)
    except Exception as e:
        logger.error("query_failed", sql=sql, error=str(e))
        return f"查询失败: {str(e)}"
```

### 4. 日志记录

记录工具调用便于调试：

```python
from app.core.logging import get_logger

logger = get_logger(__name__)

@tool
async def my_tool(input: str) -> str:
    logger.info("my_tool_invoked", input=input)
    ...
```

## 工具分类

建议按功能分类组织工具：

```
tools/
├── builtin/          # 内置通用工具
├── database/         # 数据库相关工具
├── web/             # Web/API 工具
├── file/            # 文件操作工具
└── custom/          # 自定义工具
```

## 测试工具

```python
import pytest
from app.core.agent.tools import get_tool

@pytest.mark.asyncio
async def test_my_custom_tool():
    tool = get_tool("my_custom_tool")
    assert tool is not None

    result = await tool.ainvoke({"query": "test"})
    assert "处理结果" in result
```
