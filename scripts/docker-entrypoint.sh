#!/bin/bash
set -e

# Docker entrypoint script for Kiki Agent
# This script handles database migrations and other startup tasks

echo "Starting Kiki Agent..."
echo "Environment: ${APP_ENV:-production}"
echo "Python version: $(python --version)"

# Wait for dependencies if WAIT_FOR_DEPS is set
if [ -n "$WAIT_FOR_DEPS" ]; then
    echo "Waiting for dependencies..."

    # Wait for PostgreSQL
    if [ -n "$DATABASE_URL" ]; then
        echo "Waiting for PostgreSQL..."
        host=$(echo "$DATABASE_URL" | grep -oP '(?://)[^:]+' | sed 's|//||' | cut -d'@' -f2 | cut -d':' -f1)
        port=$(echo "$DATABASE_URL" | grep -oP ':[0-9]+' | head -1 | sed 's|:||')
        host=${host:-localhost}
        port=${port:-5432}

        until nc -z "$host" "$port" 2>/dev/null; do
            echo "PostgreSQL is unavailable at $host:$port - sleeping"
            sleep 2
        done
        echo "PostgreSQL is up!"
    fi

    # Wait for Redis
    if [ -n "$REDIS_URL" ]; then
        echo "Waiting for Redis..."
        host=$(echo "$REDIS_URL" | grep -oP '(?://)[^:]+' | sed 's|//||' | cut -d'@' -f1 | cut -d':' -f1)
        port=$(echo "$REDIS_URL" | grep -oP ':[0-9]+' | head -1 | sed 's|:||')
        host=${host:-localhost}
        port=${port:-6379}

        until nc -z "$host" "$port" 2>/dev/null; do
            echo "Redis is unavailable at $host:$port - sleeping"
            sleep 2
        done
        echo "Redis is up!"
    fi
fi

# Run database migrations (if enabled)
if [ "$RUN_MIGRATIONS" = "true" ]; then
    echo "Running database migrations..."
    python -m alembic upgrade head || echo "Migration failed or not configured"
fi

# Create necessary directories
mkdir -p /app/logs
mkdir -p /app/data/files

# Execute the main command
exec "$@"
