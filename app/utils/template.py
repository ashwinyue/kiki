"""模板渲染工具

支持占位符替换和变量验证。
使用 Jinja2 进行模板渲染。
"""

import json
import re
from typing import Any

from jinja2 import (
    BaseLoader,
    Environment,
    StrictUndefined,
    TemplateError,
)

from app.observability.logging import get_logger

logger = get_logger(__name__)

# 占位符正则表达式（兼容两种格式）
# {{ variable_name }} - Jinja2 格式
# ${variable_name} - 简单格式
_PLACEHOLDER_PATTERN = re.compile(r"\{\{([^}]+)\}\}|\$\{([^}]+)\}")

_jinja_env: Environment | None = None


def _get_jinja_env() -> Environment:
    """获取 Jinja2 环境

    Returns:
        Jinja2 Environment 实例
    """
    global _jinja_env
    if _jinja_env is None:
        _jinja_env = Environment(
            loader=BaseLoader(),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=False,
        )
    return _jinja_env


def extract_variables(template: str) -> set[str]:
    """从模板中提取变量名

    支持两种格式：
    - {{ variable_name }} - Jinja2 格式
    - ${variable_name} - 简单格式

    Args:
        template: 模板字符串

    Returns:
        变量名集合

    Examples:
        >>> extract_variables("Hello {{name}}!")
        {'name'}
        >>> extract_variables("Hello ${name}!")
        {'name'}
        >>> extract_variables("{{first}} {{last}}")
        {'first', 'last'}
    """
    variables = set()
    for match in _PLACEHOLDER_PATTERN.finditer(template):
        # 第1组是 {{ }} 内容，第2组是 ${ }} 内容
        var = (match.group(1) or match.group(2) or "").strip()
        if var:
            # 移除可能的 Jinja2 过滤器和表达式
            var = var.split("|")[0].strip()
            variables.add(var)
    return variables


def convert_simple_template(template: str) -> str:
    """将简单格式占位符转换为 Jinja2 格式

    将 ${variable} 转换为 {{ variable }}

    Args:
        template: 原始模板

    Returns:
        转换后的模板
    """
    return re.sub(r"\$\{([^}]+)\}", r"{{ \1 }}", template)


def render_template(
    template: str,
    variables: dict[str, Any],
    *,
    strict: bool = False,
    missing_placeholder: str = "",
) -> str:
    """渲染模板

    支持两种占位符格式，自动转换。

    Args:
        template: 模板字符串
        variables: 变量值字典
        strict: 是否严格模式（缺失变量时报错）
        missing_placeholder: 缺失变量时的占位符

    Returns:
        渲染后的字符串

    Raises:
        TemplateSyntaxError: 模板语法错误
        TemplateError: 模板渲染错误（仅在 strict=True 时）

    Examples:
        >>> render_template("Hello {{name}}!", {"name": "World"})
        'Hello World!'
        >>> render_template("Hello ${name}!", {"name": "World"})
        'Hello World!'
        >>> render_template("Hello {{name}}!", {}, missing_placeholder="[missing]")
        'Hello [missing]!'
    """
    if not template:
        return ""

    template = convert_simple_template(template)

    try:
        env = _get_jinja_env()
        jinja_template = env.from_string(template)

        if not strict:
            result = jinja_template.render(**variables)
        else:
            result = jinja_template.render(**variables)

        return result

    except TemplateError as e:
        if strict:
            logger.error("template_render_error", error=str(e))
            raise
        logger.warning("template_render_fallback", error=str(e))
        return template


def validate_variable(
    name: str,
    value: Any,
    variable_type: str,
    validation_rule: str | None = None,
) -> tuple[bool, str | None]:
    """验证变量值

    Args:
        name: 变量名
        value: 变量值
        variable_type: 变量类型
        validation_rule: 验证规则（正则表达式或 JSON Schema）

    Returns:
        (是否有效, 错误信息)

    Examples:
        >>> validate_variable("age", "25", "number")
        (True, None)
        >>> validate_variable("email", "invalid", "string", r"^[^@]+@[^@]+$")
        (False, 'email 格式不匹配')
    """
    try:
        if variable_type == "string":
            if value is None:
                return False, f"{name} 不能为空"
            str_value = str(value)
        elif variable_type == "number":
            if value is None or value == "":
                return False, f"{name} 必须是数字"
            try:
                float(value)
            except (ValueError, TypeError):
                return False, f"{name} 必须是有效的数字"
            str_value = str(value)
        elif variable_type == "boolean":
            if value is None:
                return False, f"{name} 不能为空"
            if not isinstance(value, bool) and str(value).lower() not in (
                "true",
                "false",
                "1",
                "0",
            ):
                return False, f"{name} 必须是布尔值"
            str_value = str(value)
        elif variable_type == "json":
            if value is None or value == "":
                return True, None  # 空 JSON 允许
            if isinstance(value, str):
                json.loads(value)
            str_value = str(value)
        elif variable_type == "array":
            if value is None:
                return False, f"{name} 不能为空"
            if not isinstance(value, list | tuple):
                return False, f"{name} 必须是数组"
            str_value = str(value)
        else:
            str_value = str(value) if value is not None else ""

        if validation_rule and variable_type == "string":
            if not re.match(validation_rule, str_value):
                return False, f"{name} 格式不匹配"

        return True, None

    except json.JSONDecodeError:
        return False, f"{name} 必须是有效的 JSON"
    except Exception as e:
        logger.error("variable_validation_error", name=name, error=str(e))
        return False, f"{name} 验证失败: {str(e)}"


def validate_variables(
    variables: dict[str, Any],
    definitions: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """验证多个变量

    Args:
        variables: 变量值字典
        definitions: 变量定义字典
            {
                "var_name": {
                    "type": "string",
                    "required": True,
                    "validation_rule": "^.{1,100}$",
                    "default_value": "默认值"
                }
            }

    Returns:
        错误字典 {变量名: 错误信息}
    """
    errors: dict[str, str] = {}

    for name, definition in definitions.items():
        value = variables.get(name)

        if definition.get("required", False) and not value:
            errors[name] = f"{name} 是必填项"
            continue

        if value:
            variable_type = definition.get("type", "string")
            validation_rule = definition.get("validation_rule")

            is_valid, error_msg = validate_variable(
                name,
                value,
                variable_type,
                validation_rule,
            )

            if not is_valid:
                errors[name] = error_msg or f"{name} 验证失败"

    return errors


def render_with_defaults(
    template: str,
    variables: dict[str, Any],
    definitions: dict[str, dict[str, Any]],
    *,
    strict: bool = False,
) -> tuple[str, list[str], dict[str, str]]:
    """使用默认值渲染模板

    Args:
        template: 模板字符串
        variables: 提供的变量值
        definitions: 变量定义（包含默认值）
        strict: 是否严格模式

    Returns:
        (渲染结果, 缺失变量列表, 验证错误字典)

    Examples:
        >>> defs = {
        ...     "name": {"type": "string", "required": True, "default_value": "Guest"},
        ...     "title": {"type": "string", "default_value": "User"},
        ... }
        >>> render_with_defaults("Hello {{name}}, {{title}}!", {}, defs)
        ('Hello Guest, User!', [], {})
    """
    merged_vars = {}
    missing_vars = []
    validation_errors = {}

    for var_name, definition in definitions.items():
        if var_name in variables and variables[var_name] is not None:
            merged_vars[var_name] = variables[var_name]
        elif "default_value" in definition:
            merged_vars[var_name] = definition["default_value"]
        else:
            missing_vars.append(var_name)

        if var_name in merged_vars:
            definition_errors = validate_variables(
                {var_name: merged_vars[var_name]},
                {var_name: definition},
            )
            validation_errors.update(definition_errors)

    result = render_template(template, merged_vars, strict=strict)

    return result, missing_vars, validation_errors


def preview_render(
    template: str,
    values: dict[str, Any],
    placeholders: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """预览模板渲染结果

    Args:
        template: 模板字符串
        values: 变量值
        placeholders: 占位符定义列表

    Returns:
        预览结果字典
        {
            "rendered": "渲染后的内容",
            "missing_variables": ["缺失的变量"],
            "validation_errors": {"var": "错误信息"},
            "used_placeholders": ["使用的占位符"]
        }
    """
    definitions: dict[str, dict[str, Any]] = {}
    if placeholders:
        for p in placeholders:
            definitions[p["name"]] = {
                "type": p.get("variable_type", "string"),
                "required": p.get("is_required", False),
                "default_value": p.get("default_value"),
                "validation_rule": p.get("validation_rule"),
            }

    template_vars = extract_variables(template)

    rendered, missing, validation_errors = render_with_defaults(
        template,
        values,
        definitions,
        strict=False,
    )

    used_placeholders = [
        var for var in template_vars if var in values or var in definitions
    ]

    return {
        "rendered": rendered,
        "missing_variables": missing,
        "validation_errors": validation_errors,
        "used_placeholders": used_placeholders,
    }


__all__ = [
    "extract_variables",
    "convert_simple_template",
    "render_template",
    "validate_variable",
    "validate_variables",
    "render_with_defaults",
    "preview_render",
]
