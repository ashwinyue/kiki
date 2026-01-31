#!/bin/bash
# Kiki 数据库重建脚本
# 使用 WeKnora99 表结构
#
# 使用方式:
#   ./scripts/rebuild_db.sh [database_name] [postgres_user]
#
# 示例:
#   ./scripts/rebuild_db.sh kiki_db postgres

set -e  # 遇到错误立即退出

# 默认参数
DB_NAME=${1:-kiki_db}
PG_USER=${2:-postgres}
PG_PORT=${PG_PORT:-5432}
PG_HOST=${PG_HOST:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SQL_FILE="$SCRIPT_DIR/public.sql"

echo "========================================"
echo "Kiki 数据库重建脚本"
echo "========================================"
echo "数据库: $DB_NAME"
echo "用户: $PG_USER"
echo "主机: $PG_HOST:$PG_PORT"
echo ""

# 确认操作
read -p "⚠️  这将删除并重建数据库 '$DB_NAME'，确认继续？[yes/NO] " confirm
if [[ "$confirm" != "yes" ]]; then
    echo "❌ 操作已取消"
    exit 1
fi

echo ""
echo "📋 步骤 1/5: 删除旧数据库..."
dropdb --host="$PG_HOST" --port="$PG_PORT" --username="$PG_USER" --if-exists "$DB_NAME" 2>/dev/null || echo "  (数据库不存在，跳过)"

echo "✅ 步骤 2/5: 创建新数据库..."
createdb --host="$PG_HOST" --port="$PG_PORT" --username="$PG_USER" "$DB_NAME"

echo "✅ 步骤 3/5: 安装扩展..."
psql --host="$PG_HOST" --port="$PG_PORT" --username="$PG_USER" -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";"
psql --host="$PG_HOST" --port="$PG_PORT" --username="$PG_USER" -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>/dev/null || echo "  (vector 扩展需要 pgvector，请手动安装)"

echo "✅ 步骤 4/5: 导入表结构..."
if [[ -f "$SQL_FILE" ]]; then
    psql --host="$PG_HOST" --port="$PG_PORT" --username="$PG_USER" -d "$DB_NAME" -f "$SQL_FILE" > /dev/null
    echo "  已导入 $SQL_FILE"
else
    echo "❌ 错误: 找不到 SQL 文件 $SQL_FILE"
    exit 1
fi

echo "✅ 步骤 5/5: 验证表结构..."
TABLES=$(psql --host="$PG_HOST" --port="$PG_PORT" --username="$PG_USER" -d "$DB_NAME" -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE' AND table_name NOT LIKE 'pg_%';")
echo "  已创建 $TABLES 张表"

echo ""
echo "========================================"
echo "✅ 数据库重建完成！"
echo "========================================"
echo ""
echo "连接命令:"
echo "  psql -h $PG_HOST -p $PG_PORT -U $PG_USER -d $DB_NAME"
echo ""
