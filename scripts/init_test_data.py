#!/usr/bin/env python3
"""测试数据初始化脚本

为 E2E 测试准备测试数据库数据。

用法:
    uv run python scripts/init_test_data.py --help
    uv run python scripts/init_test_data.py init --env testing
    uv run python scripts/init_test_data.py reset --env testing
"""

import argparse
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import Settings, get_settings
from app.models.database import ChatSession, Memory, Message, Thread, User
from app.observability.logging import get_logger, set_logging_level

logger = get_logger(__name__)


# ============== 数据库配置 ==============

DATABASE_URLS = {
    "testing": "postgresql+asyncpg://postgres:postgres123!@#@localhost:15432/kiki_test",
    "development": "postgresql+asyncpg://postgres:postgres123!@#@localhost:15432/kiki",
    "production": None,  # 从环境变量读取
}


def get_database_url(env: str) -> str:
    """获取数据库连接 URL"""
    if env == "production":
        import os

        url = os.getenv("KIKI_DATABASE_URL")
        if not url:
            raise ValueError("生产环境需要设置 KIKI_DATABASE_URL 环境变量")
        return url
    return DATABASE_URLS[env]


# ============== 测试数据定义 ==============

TEST_USERS = [
    {
        "email": "test_user_1@example.com",
        "password": "TestPass123!",
        "full_name": "测试用户一",
        "is_active": True,
        "is_superuser": False,
    },
    {
        "email": "test_user_2@example.com",
        "password": "TestPass123!",
        "full_name": "测试用户二",
        "is_active": True,
        "is_superuser": False,
    },
    {
        "email": "test_admin@example.com",
        "password": "AdminPass123!",
        "full_name": "测试管理员",
        "is_active": True,
        "is_superuser": True,
    },
]


TEST_SESSIONS = [
    {"name": "产品咨询", "user_email": "test_user_1@example.com"},
    {"name": "技术支持", "user_email": "test_user_1@example.com"},
    {"name": "售后服务", "user_email": "test_user_2@example.com"},
]


TEST_MESSAGES = [
    {
        "session_name": "产品咨询",
        "role": "user",
        "content": "你们的产品价格是多少？",
    },
    {
        "session_name": "产品咨询",
        "role": "assistant",
        "content": "我们的产品有多个版本，基础版免费，专业版每月 $99。",
    },
    {
        "session_name": "产品咨询",
        "role": "user",
        "content": "专业版有什么功能？",
    },
    {
        "session_name": "技术支持",
        "role": "user",
        "content": "怎么重置密码？",
    },
    {
        "session_name": "技术支持",
        "role": "assistant",
        "content": "您可以在登录页面点击「忘记密码」链接来重置密码。",
    },
]


# ============== 初始化函数 ==============


async def create_tables(engine) -> None:
    """创建数据库表"""
    from app.models.database import SQLModel
    from sqlmodel import SQLModel as BaseModel

    logger.info("Creating database tables...")

    async with engine.begin() as conn:
        await conn.run_sync(BaseModel.metadata.create_all)

    logger.info("Database tables created successfully")


async def drop_tables(engine) -> None:
    """删除数据库表"""
    from app.models.database import SQLModel
    from sqlmodel import SQLModel as BaseModel

    logger.warning("Dropping database tables...")

    async with engine.begin() as conn:
        await conn.run_sync(BaseModel.metadata.drop_all)

    logger.info("Database tables dropped")


async def create_test_data(session: AsyncSession) -> None:
    """创建测试数据"""
    logger.info("Creating test data...")

    # ========== 创建用户 ==========
    from app.models.database import UserCreate
    from app.repositories.base import Repository

    user_map = {}
    for user_data in TEST_USERS:
        repo_user_data = UserCreate(
            email=user_data["email"],
            password=user_data["password"],
            full_name=user_data["full_name"],
        )

        from app.services.database import user_repository

        repo = user_repository(session)
        user = await repo.create_with_password(repo_user_data)

        if user_data.get("is_superuser"):
            user.is_superuser = True
            await session.commit()
            await session.refresh(user)

        user_map[user.email] = user
        logger.info(f"Created user: {user.email}")

    # ========== 创建会话 ==========
    session_map = {}
    for session_data in TEST_SESSIONS:
        user = user_map[session_data["user_email"]]

        from app.models.database import SessionCreate
        from app.services.database import session_repository

        repo = session_repository(session)
        session_obj = await repo.create_with_user(
            data=SessionCreate(name=session_data["name"]),
            user_id=user.id,
        )

        session_map[session_data["name"]] = session_obj
        logger.info(f"Created session: {session_data['name']} for user {user.email}")

    # ========== 创建消息 ==========
    for msg_data in TEST_MESSAGES:
        session_obj = session_map[msg_data["session_name"]]

        message = Message(
            role=msg_data["role"],
            content=msg_data["content"],
            session_id=session_obj.id,
        )
        session.add(message)
        logger.info(f"Created message in session: {msg_data['session_name']}")

    await session.commit()
    logger.info("Test data created successfully")


async def clear_test_data(session: AsyncSession) -> None:
    """清除测试数据"""
    logger.warning("Clearing test data...")

    # 按依赖顺序删除
    await session.exec_driver_sql("DELETE FROM messages")
    await session.exec_driver_sql("DELETE FROM chatsessions")
    await session.exec_driver_sql("DELETE FROM threads")
    await session.exec_driver_sql("DELETE FROM memories")
    await session.exec_driver_sql("DELETE FROM users")

    await session.commit()
    logger.info("Test data cleared")


# ============== 命令行接口 ==============


async def init_command(env: str, reset: bool = False) -> None:
    """初始化测试数据"""
    database_url = get_database_url(env)
    logger.info(f"Using database: {env}")

    engine = create_async_engine(database_url, echo=False)

    if reset:
        await drop_tables(engine)

    await create_tables(engine)

    async_session = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session() as session:
        await create_test_data(session)

    await engine.dispose()


async def reset_command(env: str) -> None:
    """重置测试数据"""
    database_url = get_database_url(env)
    logger.info(f"Using database: {env}")

    engine = create_async_engine(database_url, echo=False)

    async_session = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session() as session:
        await clear_test_data(session)
        await create_test_data(session)

    await engine.dispose()


async def status_command(env: str) -> None:
        """查看测试数据状态"""
        database_url = get_database_url(env)
        engine = create_async_engine(database_url, echo=False)

        async_session = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )

        async with async_session() as session:
            from sqlalchemy import text

            # 统计各表数据量
            result = await session.execute(text("SELECT COUNT(*) FROM users"))
            user_count = result.scalar()
            logger.info(f"Users count: {user_count}")

            result = await session.execute(text("SELECT COUNT(*) FROM chatsessions"))
            session_count = result.scalar()
            logger.info(f"Sessions count: {session_count}")

            result = await session.execute(text("SELECT COUNT(*) FROM messages"))
            message_count = result.scalar()
            logger.info(f"Messages count: {message_count}")

        await engine.dispose()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="测试数据初始化脚本")
    parser.add_argument(
        "--env",
        choices=["testing", "development", "production"],
        default="testing",
        help="目标环境",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="详细输出",
    )

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # init 子命令
    init_parser = subparsers.add_parser("init", help="初始化测试数据")
    init_parser.add_argument(
        "--reset",
        action="store_true",
        help="重置数据库（删除所有表后重建）",
    )

    # reset 子命令
    subparsers.add_parser("reset", help="重置测试数据")

    # status 子命令
    subparsers.add_parser("status", help="查看测试数据状态")

    # clear 子命令
    subparsers.add_parser("clear", help="清除测试数据")

    args = parser.parse_args()

    # 设置日志级别
    if args.verbose:
        set_logging_level("DEBUG")

    if not args.command:
        parser.print_help()
        return

    # 执行命令
    if args.command == "init":
        asyncio.run(init_command(args.env, args.reset))
    elif args.command == "reset":
        asyncio.run(reset_command(args.env))
    elif args.command == "status":
        asyncio.run(status_command(args.env))
    elif args.command == "clear":
        async def _clear(env: str) -> None:
            database_url = get_database_url(env)
            engine = create_async_engine(database_url, echo=False)
            async_session = async_sessionmaker(
                engine, class_=AsyncSession, expire_on_commit=False
            )
            async with async_session() as session:
                await clear_test_data(session)
            await engine.dispose()
        asyncio.run(_clear(args.env))


if __name__ == "__main__":
    main()
