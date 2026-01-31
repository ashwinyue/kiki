# Kiki Agent Makefile
# 本地开发模式: 依赖服务用 Docker，前后端应用本地启动

.PHONY: help dev dev-deps dev-backend dev-frontend dev-all
.PHONY: docker-up docker-down docker-logs docker-clean
.PHONY: backend-install backend-run backend-test backend-lint backend-shell
.PHONY: frontend-install frontend-run frontend-build frontend-lint
.PHONY: db-shell redis-shell test clean
.PHONY: obs-up obs-down obs-logs obs-status kibana prometheus grafana

# 默认目标: 显示帮助
help:
	@echo "Kiki Agent 开发命令:"
	@echo ""
	@echo "  依赖服务 (Docker):"
	@echo "    make dev-deps      - 启动依赖服务 (PostgreSQL + Redis)"
	@echo "    make docker-up     - 同上，启动依赖服务"
	@echo "    make docker-down   - 停止依赖服务"
	@echo "    make docker-logs   - 查看依赖服务日志"
	@echo "    make docker-clean  - 清理依赖服务数据和容器"
	@echo ""
	@echo "  后端开发:"
	@echo "    make backend-install   - 安装 Python 依赖 (uv sync)"
	@echo "    make backend-run       - 启动后端服务 (uvicorn)"
	@echo "    make dev-backend       - 同上"
	@echo "    make kill-port-8000    - 检测并释放端口 8000"
	@echo "    make backend-test      - 运行后端测试"
	@echo "    make backend-lint      - 代码检查 (ruff + mypy)"
	@echo ""
	@echo "  前端开发:"
	@echo "    make frontend-install  - 安装前端依赖 (pnpm install)"
	@echo "    make frontend-run      - 启动前端服务 (vite)"
	@echo "    make dev-frontend      - 同上"
	@echo "    make frontend-build    - 构建前端"
	@echo "    make frontend-lint     - 前端代码检查"
	@echo ""
	@echo "  快速启动:"
	@echo "    make dev           - 启动依赖服务 + 后端"
	@echo "    make dev-all       - 启动依赖服务 + 前后端"
	@echo ""
	@echo "  数据库操作:"
	@echo "    make db-shell      - 进入 PostgreSQL shell"
	@echo "    make redis-shell   - 进入 Redis shell"
	@echo ""
	@echo "  其他:"
	@echo "    make test          - 运行所有测试"
	@echo "    make clean         - 清理所有"
	@echo ""
	@echo "  可观测性:"
	@echo "    make obs-up        - 启动可观测性服务 (ELK + Prometheus + Grafana)"
	@echo "    make obs-down      - 停止可观测性服务"
	@echo "    make obs-logs      - 查看可观测性服务日志"
	@echo "    make obs-status    - 查看可观测性服务状态"
	@echo "    make kibana        - 打开 Kibana (http://localhost:5601)"
	@echo "    make prometheus    - 打开 Prometheus (http://localhost:9090)"
	@echo "    make grafana       - 打开 Grafana (http://localhost:3000)"

# ========== 依赖服务 (Docker) ==========

# 启动依赖服务
dev-deps: docker-up

docker-up:
	@echo "启动依赖服务..."
	docker-compose -f docker-compose.dev.yml up -d db redis
	@echo "等待服务就绪..."
	@sleep 3
	@echo "✓ 依赖服务已启动:"
	@echo "  PostgreSQL: localhost:5432"
	@echo "  Redis:      localhost:6379"

# 启动含 MinIO 的依赖服务
docker-up-storage:
	docker-compose -f docker-compose.dev.yml --profile storage up -d
	@echo "✓ 依赖服务已启动 (含 MinIO):"
	@echo "  PostgreSQL: localhost:5432"
	@echo "  Redis:      localhost:6379"
	@echo "  MinIO:      http://localhost:9000"

# 停止依赖服务
docker-down:
	docker-compose -f docker-compose.dev.yml down

# 查看依赖服务日志
docker-logs:
	docker-compose -f docker-compose.dev.yml logs -f

# 清理依赖服务
docker-clean:
	docker-compose -f docker-compose.dev.yml down -v
	@echo "✓ 已清理依赖服务数据和容器"

# ========== 后端开发 ==========

# 检查并释放端口 8000
kill-port-8000:
	@echo "检查端口 8000..."
	@PID=$$(lsof -ti:8000 2>/dev/null); \
	if [ -n "$$PID" ]; then \
		echo "端口 8000 被进程 $$PID 占用，正在终止..."; \
		kill -9 $$PID 2>/dev/null || true; \
		sleep 1; \
		echo "✓ 端口 8000 已释放"; \
	else \
		echo "✓ 端口 8000 空闲"; \
	fi

# 安装后端依赖
backend-install:
	@echo "安装 Python 依赖..."
	uv sync --all
	@echo "✓ 依赖安装完成"

# 启动后端服务
backend-run: kill-port-8000 dev-deps
	@echo "启动后端服务..."
	@sleep 2
	uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

dev-backend: backend-run

# 运行后端测试
backend-test:
	uv run pytest tests/ -v

# 运行单元测试（新模块）
test-memory-retry:
	uv run pytest tests/unit/test_memory.py tests/unit/test_retry.py -v

# 代码检查
backend-lint:
	@echo "运行 ruff 检查..."
	uv run ruff check app/ tests/
	@echo "运行 ruff 格式化检查..."
	uv run ruff format --check app/ tests/
	@echo "运行 mypy 类型检查..."
	uv run mypy app/

# 自动修复代码格式
backend-format:
	uv run ruff check --fix app/ tests/
	uv run ruff format app/ tests/

# 数据库 shell
db-shell:
	docker-compose -f docker-compose.dev.yml exec db psql -U kiki -d kiki

# Redis shell
redis-shell:
	docker-compose -f docker-compose.dev.yml exec redis redis-cli

# ========== 前端开发 ==========

# 安装前端依赖
frontend-install:
	@echo "安装前端依赖..."
	cd frontend && pnpm install
	@echo "✓ 前端依赖安装完成"

# 启动前端服务
frontend-run:
	@echo "启动前端服务..."
	cd frontend && pnpm dev

dev-frontend: frontend-run

# 构建前端
frontend-build:
	@echo "构建前端..."
	cd frontend && pnpm build

# 前端代码检查
frontend-lint:
	cd frontend && pnpm lint

# ========== 快速启动 ==========

# 启动依赖 + 后端
dev: kill-port-8000 dev-deps
	@echo "等待数据库就绪..."
	@sleep 3
	@echo "启动后端服务..."
	uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 启动所有 (需要另开终端启动前端)
dev-all: dev-deps
	@echo ""
	@echo "✓ 依赖服务已启动"
	@echo ""
	@echo "请另开终端运行:"
	@echo "  make dev-backend   # 启动后端"
	@echo "  make dev-frontend  # 启动前端"
	@echo ""
	@echo "或使用 tmux 分屏:"
	@echo "  tmux new -s kiki"
	@echo "  # Ctrl+B + % 创建分屏"
	@echo "  # 左屏: make dev-backend"
	@echo "  # 右屏: make dev-frontend"

# ========== 其他 ==========

# 运行所有测试
test:
	uv run pytest tests/ -v

# 清理所有
clean: docker-clean
	@echo "清理本地缓存..."
	rm -rf frontend/node_modules frontend/dist frontend/.vite
	rm -rf .venv uv.lock
	@echo "✓ 清理完成"

# ========== 可观测性 ==========

# 启动可观测性服务栈
obs-up:
	@echo "启动可观测性服务栈..."
	docker-compose -f docker-compose.observability.yml up -d
	@echo "等待服务启动..."
	@sleep 10
	@echo "✓ 可观测性服务已启动:"
	@echo "  Elasticsearch: http://localhost:9200"
	@echo "  Kibana:        http://localhost:5601"
	@echo "  Prometheus:    http://localhost:9090"
	@echo "  Grafana:       http://localhost:3000 (admin/admin)"
	@echo "  Alertmanager:  http://localhost:9093"

# 停止可观测性服务
obs-down:
	docker-compose -f docker-compose.observability.yml down
	@echo "✓ 可观测性服务已停止"

# 查看可观测性服务日志
obs-logs:
	docker-compose -f docker-compose.observability.yml logs -f

# 查看可观测性服务状态
obs-status:
	@echo "可观测性服务状态:"
	@docker-compose -f docker-compose.observability.yml ps

# 打开 Kibana
kibana:
	@echo "正在打开 Kibana..."
	@if command -v open >/dev/null 2>&1; then \
		open http://localhost:5601; \
	elif command -v xdg-open >/dev/null 2>&1; then \
		xdg-open http://localhost:5601; \
	else \
		echo "请手动打开: http://localhost:5601"; \
	fi

# 打开 Prometheus
prometheus:
	@echo "正在打开 Prometheus..."
	@if command -v open >/dev/null 2>&1; then \
		open http://localhost:9090; \
	elif command -v xdg-open >/dev/null 2>&1; then \
		xdg-open http://localhost:9090; \
	else \
		echo "请手动打开: http://localhost:9090"; \
	fi

# 打开 Grafana
grafana:
	@echo "正在打开 Grafana..."
	@if command -v open >/dev/null 2>&1; then \
		open http://localhost:3000; \
	elif command -v xdg-open >/dev/null 2>&1; then \
		xdg-open http://localhost:3000; \
	else \
		echo "请手动打开: http://localhost:3000 (admin/admin)"; \
	fi
