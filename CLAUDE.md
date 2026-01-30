# Kiki Agent Framework

## 项目规约

本项目在 `.claude/rules/` 目录下定义了强制性的开发规约，所有 AI 辅助开发**必须**遵守：

| 规约文件 | 说明 |
|---------|------|
| `python-package-management.md` | **强制**: 仅使用 uv 管理 Python 依赖 |

## 技术栈

- **Web**: FastAPI + Uvicorn
- **Agent**: LangGraph + LangChain
- **数据库**: PostgreSQL (asyncpg) + SQLModel
- **可观测性**: structlog + Langfuse + Prometheus
- **测试**: pytest + pytest-asyncio + pytest-cov
- **代码质量**: ruff + mypy

## 项目结构

```
kiki/
├── app/
│   ├── api/           # API 路由层
│   ├── core/          # 核心模块 (config, logging, middleware, agent)
│   ├── models/        # 数据模型
│   ├── schemas/       # Pydantic 模式
│   ├── services/      # 业务服务
│   ├── repositories/  # 数据访问
│   ├── utils/         # 工具函数
│   └── main.py        # 应用入口
├── tests/             # 测试
├── .claude/           # AI 上下文配置
│   ├── rules/         # 开发规约 (强制遵守)
│   └── skills/        # 项目技能定义
├── pyproject.toml     # 项目配置 (使用 uv)
└── uv.lock            # 依赖锁文件 (必须提交)
```

## 快速命令

```bash
# 安装依赖
uv sync

# 运行开发服务器
uv run uvicorn app.main:app --reload

# 运行测试
uv run pytest

# 代码检查
uv run ruff check .
uv run mypy app/
```
