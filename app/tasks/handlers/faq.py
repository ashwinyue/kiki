"""FAQ 导入任务处理器

对齐 WeKnora99 的 FAQ 导入功能 (TypeFAQImport)
"""

import re
import uuid
from datetime import UTC, datetime

from app.infra.database import async_session_factory
from app.models.faq import FAQ, FAQCategory, FAQStatus
from app.observability.logging import get_logger
from app.tasks.handlers.base import ProgressReporter, TaskHandler, task_context

logger = get_logger(__name__)


async def process_faq_import(
    celery_task,
    payload: dict,
    tenant_id: int,
) -> dict:
    """处理 FAQ 导入任务

    Args:
        celery_task: Celery 任务实例
        payload: 任务参数
        tenant_id: 租户 ID

    Returns:
        处理结果
    """
    task_id = payload.get("task_id")
    kb_id = payload.get("kb_id")
    entries = payload.get("entries", [])
    dry_run = payload.get("dry_run", False)

    async with task_context(celery_task, task_id, tenant_id) as handler:
        async with async_session_factory() as session:
            # 创建任务记录
            await handler.create_task(
                task_type="faq:import",
                payload=payload,
                title=f"FAQ 导入 ({'Dry Run' if dry_run else '正式导入'})",
                business_id=kb_id,
                business_type="knowledge_base",
                total_items=len(entries),
            )

            await handler.mark_started()

            try:
                # 如果是 dry run，只验证不导入
                if dry_run:
                    await handler.update_progress(50, "验证 FAQ 条目")

                    validation_results = await _validate_faq_entries(entries)
                    valid_count = sum(1 for r in validation_results if r["is_valid"])
                    invalid_count = len(validation_results) - valid_count

                    await handler.update_progress(100, "验证完成")

                    result = {
                        "status": "completed",
                        "dry_run": True,
                        "total_entries": len(entries),
                        "valid_count": valid_count,
                        "invalid_count": invalid_count,
                        "validation_details": validation_results,
                    }

                    await handler.mark_completed(result)
                    return result

                # 正式导入
                await handler.update_progress(10, "开始导入 FAQ 条目")

                imported_count = 0
                skipped_count = 0
                error_count = 0
                reporter = ProgressReporter(handler, total=len(entries), current_step="导入 FAQ")

                for i, entry in enumerate(entries):
                    await handler.update_progress(
                        10 + int((i / len(entries)) * 80),
                        f"导入: {entry.get('question', '')[:30]}",
                        processed_items=i + 1,
                        total_items=len(entries),
                    )

                    try:
                        # 验证条目
                        validation = _validate_single_entry(entry)
                        if not validation["is_valid"]:
                            skipped_count += 1
                            reporter.failed += 1
                            logger.warning(
                                "faq_entry_invalid",
                                entry=entry,
                                errors=validation.get("errors"),
                            )
                            continue

                        # 导入条目
                        faq = await _import_faq_entry(
                            session,
                            entry,
                            kb_id,
                            tenant_id,
                            payload.get("created_by"),
                        )
                        imported_count += 1
                        reporter.processed += 1

                    except Exception as e:
                        error_count += 1
                        reporter.failed += 1
                        logger.error(
                            "faq_import_failed",
                            entry=entry,
                            error=str(e),
                        )
                        continue

                result = {
                    "status": "completed",
                    "dry_run": False,
                    "total_entries": len(entries),
                    "imported_count": imported_count,
                    "skipped_count": skipped_count,
                    "error_count": error_count,
                }

                await handler.mark_completed(result)
                return result

            except Exception as e:
                import traceback

                await handler.mark_failed(str(e), traceback.format_exc())
                raise


def _validate_single_entry(entry: dict) -> dict:
    """验证单个 FAQ 条目

    Args:
        entry: FAQ 条目数据

    Returns:
        验证结果
    """
    errors = []
    is_valid = True

    # 验证必填字段
    if not entry.get("question"):
        errors.append("question 不能为空")
        is_valid = False
    elif len(entry["question"]) < 5:
        errors.append("question 长度至少 5 个字符")
        is_valid = False

    if not entry.get("answer"):
        errors.append("answer 不能为空")
        is_valid = False
    elif len(entry["answer"]) < 10:
        errors.append("answer 长度至少 10 个字符")
        is_valid = False

    # 验证 question 重复（简单检查）
    question = entry.get("question", "").strip()
    if question and len(question) > 200:
        errors.append("question 长度不能超过 200 个字符")
        is_valid = False

    # 验证 category
    category = entry.get("category")
    if category and category not in FAQCategory.__members__.values():
        errors.append(f"category 无效，可选值: {list(FAQCategory.__members__.values())}")
        is_valid = False

    return {
        "is_valid": is_valid,
        "errors": errors,
        "entry": entry,
    }


async def _validate_faq_entries(entries: list[dict]) -> list[dict]:
    """批量验证 FAQ 条目

    Args:
        entries: FAQ 条目列表

    Returns:
        验证结果列表
    """
    results = []
    for entry in entries:
        results.append(_validate_single_entry(entry))
    return results


async def _import_faq_entry(
    session,
    entry: dict,
    kb_id: str,
    tenant_id: int,
    created_by: str | None = None,
) -> FAQ:
    """导入单个 FAQ 条目

    Args:
        session: 数据库会话
        entry: FAQ 条目数据
        kb_id: 知识库 ID
        tenant_id: 租户 ID
        created_by: 创建者

    Returns:
        创建的 FAQ 对象
    """
    now = datetime.now(UTC)

    # 生成 slug
    slug = _generate_slug(entry.get("question", ""))

    # 确定状态
    status = FAQStatus(entry.get("status", "published"))

    # 确定分类
    category = FAQCategory(entry.get("category", "general"))

    # 解析标签
    tags = entry.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]

    faq = FAQ(
        id=str(uuid.uuid4()),
        question=entry.get("question", "").strip(),
        answer=entry.get("answer", "").strip(),
        category=category,
        tags=tags,
        priority=entry.get("priority", 0),
        locale=entry.get("locale", "zh-CN"),
        status=status,
        slug=slug,
        view_count=0,
        helpful_count=0,
        not_helpful_count=0,
        created_at=now,
        updated_at=now,
        published_at=now if status == FAQStatus.PUBLISHED else None,
        tenant_id=tenant_id,
        created_by=created_by,
        updated_by=created_by,
        knowledge_base_id=kb_id,
    )

    session.add(faq)
    await session.commit()
    await session.refresh(faq)

    logger.info(
        "faq_imported",
        faq_id=faq.id,
        question=faq.question[:50],
    )

    return faq


def _generate_slug(text: str) -> str:
    """从文本生成 slug

    Args:
        text: 原始文本

    Returns:
        slug 字符串
    """
    # 移除特殊字符，只保留中文、英文、数字
    slug = re.sub(r"[^\w\s-]", "", text.lower())
    # 替换空格为连字符
    slug = re.sub(r"[\s-]+", "-", slug)
    # 移除首尾连字符
    slug = slug.strip("-")
    # 限制长度
    slug = slug[:100]

    # 添加随机后缀避免重复
    import random
    random_suffix = random.randint(1000, 9999)
    return f"{slug}-{random_suffix}"
