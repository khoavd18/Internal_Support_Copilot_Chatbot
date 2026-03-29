from __future__ import annotations

import re
from typing import Any


# GitHub Docs / Markdown cleanup patterns
GITHUB_TEMPLATE_PATTERN = re.compile(r"\{%\s*data.*?%\}", re.DOTALL)
GENERIC_LIQUID_TAG_PATTERN = re.compile(r"\{%\s*.*?%\}", re.DOTALL)
LIQUID_OUTPUT_PATTERN = re.compile(r"\{\{.*?\}\}", re.DOTALL)
AUTOTITLE_LINK_PATTERN = re.compile(r"\[AUTOTITLE\]\((.*?)\)")
HTML_COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)
FRONTMATTER_PATTERN = re.compile(r"^---\s*\n.*?\n---\s*\n?", re.DOTALL)
MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
MULTIPLE_NEWLINES_PATTERN = re.compile(r"\n{3,}")
MULTIPLE_SPACES_PATTERN = re.compile(r"[ \t]{2,}")


def clean_text(text: str) -> str:
    """Làm sạch text trước khi chunking / embedding."""
    if not text:
        return ""

    text = FRONTMATTER_PATTERN.sub("", text)
    text = GITHUB_TEMPLATE_PATTERN.sub(" ", text)
    text = GENERIC_LIQUID_TAG_PATTERN.sub(" ", text)
    text = LIQUID_OUTPUT_PATTERN.sub(" ", text)
    text = HTML_COMMENT_PATTERN.sub(" ", text)
    text = AUTOTITLE_LINK_PATTERN.sub(r"\1", text)
    text = MARKDOWN_LINK_PATTERN.sub(r"\1", text)

    text = MULTIPLE_SPACES_PATTERN.sub(" ", text)
    text = MULTIPLE_NEWLINES_PATTERN.sub("\n\n", text)

    text = text.replace("\u200b", " ")
    text = text.replace("  \n", "\n")
    text = text.replace("****", "")
    text = text.replace("### ", "### ")
    text = text.replace("## ", "## ")
    text = text.replace("# ", "# ")

    return text.strip()


def first_non_empty(record: dict[str, Any], keys: list[str]) -> str:
    """Lấy giá trị đầu tiên không rỗng trong nhiều key."""
    for key in keys:
        value = record.get(key)

        if value is None:
            continue

        if isinstance(value, str) and value.strip():
            return value.strip()

        if isinstance(value, (int, float)):
            return str(value)

    return ""


def normalize_text(value: Any) -> str:
    """Chuẩn hóa nhiều kiểu dữ liệu về string."""
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(
                    first_non_empty(
                        item,
                        ["body", "text", "content", "comment", "message", "summary"],
                    )
                )
            else:
                parts.append(str(item).strip())
        return "\n".join([p for p in parts if p])

    if isinstance(value, dict):
        return first_non_empty(
            value, ["body", "text", "content", "comment", "message", "summary"]
        )

    return str(value).strip()


def join_non_empty(parts: list[str]) -> str:
    """Ghép các phần text lại, bỏ phần rỗng."""
    return "\n\n".join([p for p in parts if p and p.strip()])