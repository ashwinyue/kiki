-- ============================================
-- Kiki Agent Framework - 数据库初始化脚本
-- 数据库: PostgreSQL 15+
-- 编码: UTF-8
-- 说明: 完整的表结构初始化，包含所有业务表
-- ============================================

-- ============================================
-- 0. 扩展
-- ============================================

-- 启用 UUID 扩展（如果未启用）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 注意：时间戳自动管理由 SQLAlchemy 的 server_default + onupdate 实现
-- 无需使用数据库触发器


-- ============================================
-- 1. 枚举类型定义
-- ============================================

-- API Key 状态枚举
CREATE TYPE api_key_status AS ENUM (
    'active',
    'inactive',
    'revoked',
    'expired'
);

-- API Key 类型枚举
CREATE TYPE api_key_type AS ENUM (
    'personal',
    'service',
    'mcp',
    'webhook'
);


-- ============================================
-- 2. 用户相关表
-- ============================================

-- 2.1 用户表 (users)
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    full_name VARCHAR(255),
    hashed_password VARCHAR(255) NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    is_superuser BOOLEAN NOT NULL DEFAULT FALSE,
    tenant_id INTEGER,
    can_access_all_tenants BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_is_active ON users(is_active);
CREATE INDEX IF NOT EXISTS idx_users_tenant_id ON users(tenant_id);

-- 注释
COMMENT ON TABLE users IS '用户表';
COMMENT ON COLUMN users.email IS '用户邮箱（唯一）';
COMMENT ON COLUMN users.hashed_password IS '加密后的密码（bcrypt）';
COMMENT ON COLUMN users.tenant_id IS '关联租户 ID';
COMMENT ON COLUMN users.can_access_all_tenants IS '是否可访问所有租户';

