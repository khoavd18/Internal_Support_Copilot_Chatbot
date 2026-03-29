from __future__ import annotations

from typing import Iterable

from langchain_core.documents import Document


SEPARATOR = "\n\n" + ("=" * 80) + "\n\n"


def _clean_inline_text(text: str) -> str:
    if not text:
        return ""
    return text.strip()


def _format_source_line(doc: Document, index: int) -> str:
    title = (
        doc.metadata.get("title")
        or doc.metadata.get("path")
        or doc.metadata.get("doc_id")
        or f"Document {index}"
    )
    source = doc.metadata.get("source") or "unknown"
    path = doc.metadata.get("path") or ""
    url = doc.metadata.get("url") or ""

    parts = [f"[{index}] {title}", f"source={source}"]
    if path:
        parts.append(f"path={path}")
    if url:
        parts.append(f"url={url}")

    return " | ".join(parts)


def _format_context_block(doc: Document, index: int) -> str:
    source_line = _format_source_line(doc, index)
    content = _clean_inline_text(doc.page_content)

    return f"{source_line}\nCONTENT:\n{content}"


def format_context(documents: Iterable[Document]) -> str:
    blocks = []
    for idx, doc in enumerate(documents, start=1):
        blocks.append(_format_context_block(doc, idx))
    return SEPARATOR.join(blocks)


def build_prompt(question: str, documents: list[Document]) -> str:
    context = format_context(documents)

    return f"""Bạn là trợ lý hỗ trợ tài liệu kỹ thuật nội bộ.

Mục tiêu:
- Chỉ trả lời dựa trên CONTEXT được cung cấp.
- Không bịa thêm thông tin ngoài CONTEXT.
- Nếu CONTEXT chưa đủ chắc chắn, hãy nói rõ là chưa đủ thông tin.
- Nếu câu hỏi là follow-up đã được rewrite, hãy trả lời theo ý nghĩa của câu đã rewrite.
- Ưu tiên trả lời đúng thao tác thực tế, đúng lệnh, đúng URL, đúng tên tính năng.

Quy tắc rất quan trọng:
- Không giải thích lan man.
- Không viết các đoạn nhập môn chung chung nếu người dùng đang hỏi thao tác cụ thể.
- Không tự tạo URL giả hoặc chèn khoảng trắng sai vào URL/lệnh.
- Nếu có lệnh shell/git, đặt trong code block.
- Giữ nguyên lệnh kỹ thuật chính xác như trong tài liệu nếu tài liệu có nêu.
- Nếu đang so sánh hai cách (ví dụ HTTPS thay vì SSH), hãy nêu rõ phần cần đổi.

Format trả lời:
- Viết bằng tiếng Việt.
- Với câu hỏi how-to, dùng đúng cấu trúc này:
  Cách làm
  1. ...
  2. ...
  3. ...

  Lệnh mẫu
  ```bash
  ...
  ```

CONTEXT:
{context}

QUESTION: {question}
"""