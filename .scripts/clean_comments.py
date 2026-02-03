#!/usr/bin/env python3
"""自动化清理Python文件中的冗余注释"""

import re
from pathlib import Path
from typing import Tuple

# 要处理的目录
TARGET_DIR = Path("/Users/mervyn/PycharmProjects/kiki/app")

# 冗余注释模式
PATTERNS = [
    # 1. 显而易见的操作注释（中文）
    (r"# (获取|验证|检查|尝试|准备|执行|清理|删除|添加|创建|初始化|构建|解析|转换)[\u4e00-\u9fff\w\s]+(?!\s*\()", "显而易见的操作注释"),

    # 2. 测试步骤编号注释
    (r"#\s*\d+\.\s*测试\.?", "测试步骤编号"),

    # 3. 分隔符注释（只在特定上下文中冗余）
    (r"^# ={10,}\s*$", "过多等号的分隔符"),

    # 4. 空docstring（只有类名，无额外信息）
    (r'class\s+\w+.*:\n\s*"""[^"]{0,15}"""', "空docstring（类名重复）'),
]

def should_remove_comment(line: str, context: list[str]) -> bool:
    """判断是否应该删除注释"""
    # 跳过docstring内的行
    in_docstring = False
    for ctx_line in context[-10:]:  # 检查前10行
        if '"""' in ctx_line:
            in_docstring = not in_docstring
            continue

    # 跳过在docstring块内的行
    if in_docstring:
        return False

    # 检查是否匹配冗余模式
    for pattern, reason in PATTERNS:
        if re.search(pattern, line):
            print(f"  [{reason}] {line.strip()[:60]}")
            return True

    return False

def clean_file(file_path: Path) -> Tuple[int, int]:
    """清理单个文件"""
    try:
        content = file_path.read_text(encoding='utf-8')
        lines = content.split('\n')

        original_count = len(lines)
        cleaned_lines = []
        removed = 0

        i = 0
        while i < len(lines):
            line = lines[i]

            # 检查是否是注释行
            if re.match(r'^\s*#', line):
                # 获取上下文
                context = lines[max(0, i-5):i+2]

                # 判断是否删除
                if should_remove_comment(line, context):
                    removed += 1
                    i += 1
                    continue

            cleaned_lines.append(line)
            i += 1

        if removed > 0:
            # 保留git追踪
            file_path.write_text('\n'.join(cleaned_lines), encoding='utf-8')
            print(f"✅ {file_path.relative_to(TARGET_DIR)}: 移除 {removed} 条注释")

        return removed, original_count - removed

    except Exception as e:
        print(f"❌ {file_path.relative_to(TARGET_DIR)}: 错误 - {e}")
        return 0, 0

def main():
    """主函数"""
    print("🔍 开始扫描冗余注释...\n")

    total_removed = 0
    total_files = 0

    # 扫描所有Python文件
    for file_path in TARGET_DIR.rglob("*.py"):
        # 跳过某些目录
        if "legacy" in file_path.parts or "__pycache__" in file_path.parts:
            continue

        removed, _ = clean_file(file_path)
        if removed > 0:
            total_files += 1
            total_removed += removed

    print(f"\n📊 清理完成:")
    print(f"  - 处理文件数: {total_files}")
    print(f"  - 移除注释数: {total_removed}")
    print(f"  - 清理率: {total_removed / 1886 * 100:.1f}%")

if __name__ == "__main__":
    main()
