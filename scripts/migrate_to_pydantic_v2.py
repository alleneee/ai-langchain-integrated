#!/usr/bin/env python
"""
Pydantic v1 到 v2 迁移脚本

这个脚本帮助将项目代码中的 Pydantic v1 导入替换为 Pydantic v2 导入，
以适应 LangChain 0.3 版本的要求。

使用方法:
    python scripts/migrate_to_pydantic_v2.py [目录]

示例:
    python scripts/migrate_to_pydantic_v2.py src/
"""

import os
import re
import sys
import argparse
from typing import List, Tuple, Dict


PYDANTIC_V1_IMPORTS = [
    r"from\s+langchain_core\.pydantic_v1\s+import\s+(.*)",
    r"from\s+langchain\.pydantic_v1\s+import\s+(.*)",
    r"from\s+pydantic\.v1\s+import\s+(.*)",
]

VALIDATOR_REPLACEMENTS = {
    "validator": "field_validator",
    "root_validator": "model_validator",
}


def find_python_files(directory: str) -> List[str]:
    """查找目录中的所有Python文件"""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files


def process_file(file_path: str, dry_run: bool = False) -> Tuple[bool, Dict[str, int]]:
    """处理单个Python文件，替换Pydantic v1导入"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    original_content = content
    stats = {"imports": 0, "validators": 0}

    # 替换导入
    for pattern in PYDANTIC_V1_IMPORTS:
        matches = re.findall(pattern, content)
        if matches:
            stats["imports"] += len(matches)
            content = re.sub(pattern, r"from pydantic import \1", content)

    # 替换验证器
    for old, new in VALIDATOR_REPLACEMENTS.items():
        matches = re.findall(r"@" + old + r"\b", content)
        if matches:
            stats["validators"] += len(matches)
            content = re.sub(r"@" + old + r"\b", "@" + new, content)

    # 检查是否有model_rebuild()的需要
    if re.search(r"class\s+\w+\((Base\w+Parser|Base\w+Tool|Base\w+Model|BaseTool|BaseChatModel|LLM)\):", content) and "model_rebuild()" not in content:
        print(f"警告: {file_path} 中可能需要添加 model_rebuild() 调用")

    # 如果内容有修改，则保存
    if content != original_content:
        if not dry_run:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        return True, stats
    return False, stats


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Pydantic v1 到 v2 迁移工具")
    parser.add_argument("directory", help="要处理的目录路径")
    parser.add_argument("--dry-run", action="store_true", help="仅预览更改但不实际修改文件")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"错误: 目录 '{args.directory}' 不存在")
        sys.exit(1)

    python_files = find_python_files(args.directory)
    print(f"找到 {len(python_files)} 个Python文件需要处理")

    total_modified = 0
    total_stats = {"imports": 0, "validators": 0}

    for file_path in python_files:
        modified, stats = process_file(file_path, args.dry_run)
        if modified:
            print(f"修改了 {file_path}")
            print(f"  - 导入替换: {stats['imports']}")
            print(f"  - 验证器替换: {stats['validators']}")
            total_modified += 1
            total_stats["imports"] += stats["imports"]
            total_stats["validators"] += stats["validators"]

    print("\n总结:")
    print(f"处理了 {len(python_files)} 个文件")
    print(f"修改了 {total_modified} 个文件")
    print(f"替换了 {total_stats['imports']} 个导入语句")
    print(f"替换了 {total_stats['validators']} 个验证器")

    if args.dry_run:
        print("\n这是一次模拟运行，未实际修改任何文件。")
        print("使用不带 --dry-run 的命令来应用更改。")


if __name__ == "__main__":
    main() 