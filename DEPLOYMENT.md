# Docker 部署指南

## 快速开始

### 1. 配置环境变量

```bash
# 复制 Docker 环境配置
cp .env.docker .env

# 编辑 .env 文件，设置必要的配置
# - KIKI_LLM_API_KEY: LLM API 密钥
# - KIKI_SECRET_KEY: 认证密钥（生产环境至少 32 字符）
```

### 2. 启动服务

```bash
# 启动基础服务（数据库 + Redis + 应用）
make up
# 或
docker-compose up -d

# 启动所有服务（含监控和存储）
make up-all
# 或
docker-compose --profile storage --profile monitoring up -d
```

### 3. 访问服务

| 服务 | 地址 | 说明 |
|------|------|------|
| API | http://localhost:8000 | 应用 API |
| API 文档 | http://localhost:8000/docs | Swagger 文档 |
| PostgreSQL | localhost:5432 | 数据库 |
| Redis | localhost:6379 | 缓存 |
| MinIO | http://localhost:9000 | 对象存储（需启用） |
| Prometheus | http://localhost:9090 | 监控（需启用） |
| Grafana | http://localhost:3000 | 可视化（需启用） |

## Makefile 命令

```bash
make build          # 构建镜像
make up             # 启动基础服务
make up-all         # 启动所有服务
make down           # 停止服务
make restart        # 重启应用
make logs           # 查看所有日志
make logs-app       # 查看应用日志
make logs-db        # 查看数据库日志
make shell          # 进入应用容器
make db-shell       # 进入数据库 shell
make clean          # 清理所有容器和卷
make ps             # 查看服务状态
make stats          # 查看资源使用
```

## 生产环境部署

```bash
# 使用生产环境配置启动
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

生产环境配置包括：
- 多实例部署（2 个副本）
- Nginx 反向代理
- 资源限制
- 自动重启

## 监控

### Prometheus

访问 http://localhost:9090 查看指标：
- 请求速率
- 响应时间
- 错误率
- 资源使用

### Grafana

访问 http://localhost:3000（admin/admin）查看可视化仪表盘。

## 故障排查

### 查看日志

```bash
# 应用日志
make logs-app

# 数据库日志
make logs-db

# 所有服务日志
docker-compose logs -f [service-name]
```

### 进入容器

```bash
# 应用容器
make shell

# 数据库容器
make db-shell
```

### 重置数据

```bash
# 停止服务并清理卷
make clean

# 重新启动
make up
```

## 配置文件结构

```
config/
├── grafana/
│   └── provisioning/
│       ├── datasources/
│       └── dashboards/
├── nginx/
│   └── nginx.conf
└── prometheus/
    └── prometheus.yml
```
