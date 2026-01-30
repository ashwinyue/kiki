# Kiki Agent Framework

> 企业级 Agent 开发脚手架 - 基于 FastAPI + LangGraph

## 特性

- **LangGraph 集成**: 开箱即用的 Agent 工作流编排
- **生产就绪**: 认证、授权、可观测性完整支持
- **模块化架构**: 清晰的分层设计，易于扩展
- **类型安全**: 完整的类型注解，支持 mypy 检查

## 快速开始

```bash
# 安装依赖
pip install -e ".[dev]"

# 配置环境变量
cp .env.example .env

# 启动服务
uvicorn app.main:app --reload
```

## 项目结构

```
kiki/
├── app/
│   ├── api/           # API 路由层
│   │   └── v1/        # v1 版本 API
│   ├── core/          # 核心模块
│   │   ├── config.py  # 配置管理
│   │   ├── logging.py # 日志配置
│   │   ├── middleware.py  # 中间件
│   │   └── agent/     # Agent 相关
│   ├── models/        # 数据模型
│   ├── schemas/       # Pydantic 模式
│   ├── services/      # 业务服务
│   ├── repositories/  # 数据访问
│   ├── utils/         # 工具函数
│   └── main.py        # 应用入口
├── tests/             # 测试
├── prometheus/        # Prometheus 配置
├── grafana/           # Grafana 仪表板
└── docker-compose.yml # 开发环境
```

## 文档

- [架构设计](AGENTS.md) - 企业级架构指南
- [API 文档](docs/api.md) - API 参考
- [开发指南](docs/development.md) - 开发规范

## 许可证

MIT
