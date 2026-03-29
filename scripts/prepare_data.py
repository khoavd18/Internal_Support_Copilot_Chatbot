from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Any

from bs4 import BeautifulSoup


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

RAW_DIR = PROJECT_ROOT / "data_source" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data_source" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

GITHUB_DOCS_PREFIXES = [
    "get-started",
    "authentication",
    "repositories",
]


# =========================
# Generic helpers
# =========================
def save_jsonl(records: list[dict], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_json(data: dict[str, Any], output_path: Path) -> None:
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def collapse_blank_lines(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"[WARN] Cannot read {path}: {e}")
        return None


def load_json_file(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Cannot read {path}: {e}")
        return None


# =========================
# GitHub Docs
# =========================
def clean_markdown(text: str) -> str:
    text = normalize_newlines(text)

    # remove front matter
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.DOTALL)

    # remove common liquid include noise (optional but practical)
    text = re.sub(r"{%\s*data\s+variables\.[^%]+%}", " ", text)
    text = re.sub(r"{%[^%]*%}", " ", text)
    text = re.sub(r"{{[^}]*}}", " ", text)

    text = collapse_blank_lines(text)
    return text


def extract_title_from_markdown(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def build_github_docs_url(rel_path: str) -> str:
    # content/authentication/x.md -> https://docs.github.com/en/authentication/x
    if rel_path.endswith(".md"):
        rel_path = rel_path[:-3]
    return f"https://docs.github.com/en/{rel_path}"


def process_github_docs() -> list[dict]:
    repo_dir = RAW_DIR / "github_docs" / "repo"
    content_dir = repo_dir / "content"
    documents: list[dict] = []

    print(f"[DEBUG] repo_dir = {repo_dir}")
    print(f"[DEBUG] content_dir exists = {content_dir.exists()}")

    if not content_dir.exists():
        print(f"[SKIP] Không tìm thấy {content_dir}")
        return documents

    md_files = list(content_dir.rglob("*.md"))
    print(f"[DEBUG] total markdown files found = {len(md_files)}")

    for md_file in md_files:
        rel_path = md_file.relative_to(content_dir).as_posix()

        if not any(rel_path.startswith(prefix) for prefix in GITHUB_DOCS_PREFIXES):
            continue

        raw_text = safe_read_text(md_file)
        if raw_text is None:
            continue

        text = clean_markdown(raw_text)
        if len(text) < 300:
            continue

        title = extract_title_from_markdown(text, md_file.stem.replace("-", " "))
        section = rel_path.split("/")[0]

        documents.append(
            {
                "doc_id": f"github_docs::{rel_path}",
                "source": "github_docs",
                "source_type": "markdown",
                "doc_type": "document",
                "title": title,
                "section": section,
                "path": rel_path,
                "url": build_github_docs_url(rel_path),
                "text": text,
            }
        )

    print(f"[DONE] GitHub Docs documents: {len(documents)}")
    return documents


# =========================
# GitLab Handbook
# =========================
def clean_html_text(html: str) -> tuple[str, str]:
    html = normalize_newlines(html)
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    # Prefer main/article/body if available to reduce layout noise
    root = soup.find("main") or soup.find("article") or soup.body or soup

    text = root.get_text(separator="\n")
    text = normalize_newlines(text)
    text = collapse_blank_lines(text)

    return title, text


def load_gitlab_manifest_lookup() -> dict[str, dict]:
    manifest_path = RAW_DIR / "gitlab_handbook" / "manifest.json"
    manifest = load_json_file(manifest_path)

    if not isinstance(manifest, dict):
        return {}

    pages = manifest.get("pages", {})
    if not isinstance(pages, dict):
        return {}

    lookup: dict[str, dict] = {}

    for url, page_info in pages.items():
        if not isinstance(page_info, dict):
            continue

        file_name = str(page_info.get("file_name") or "").strip()
        if not file_name:
            continue

        lookup[file_name] = {
            "url": url,
            "title": page_info.get("title") or "",
            "fetched_at": page_info.get("fetched_at") or "",
        }

    return lookup


def process_gitlab_handbook() -> list[dict]:
    html_dir = RAW_DIR / "gitlab_handbook" / "html"
    documents: list[dict] = []

    print(f"[DEBUG] gitlab html_dir = {html_dir}")
    print(f"[DEBUG] gitlab html_dir exists = {html_dir.exists()}")

    if not html_dir.exists():
        print(f"[SKIP] Không tìm thấy {html_dir}")
        return documents

    manifest_lookup = load_gitlab_manifest_lookup()

    html_files = list(html_dir.glob("*.html"))
    print(f"[DEBUG] total html files found = {len(html_files)}")

    for html_file in html_files:
        html = safe_read_text(html_file)
        if html is None:
            continue

        title, text = clean_html_text(html)
        if len(text) < 300:
            continue

        manifest_info = manifest_lookup.get(html_file.name, {})
        final_title = title or manifest_info.get("title") or html_file.stem
        url = manifest_info.get("url", "")

        documents.append(
            {
                "doc_id": f"gitlab_handbook::{html_file.stem}",
                "source": "gitlab_handbook",
                "source_type": "html",
                "doc_type": "document",
                "title": final_title,
                "section": "policy",
                "path": html_file.name,
                "url": url,
                "text": text,
            }
        )

    print(f"[DONE] GitLab Handbook documents: {len(documents)}")
    return documents


# =========================
# GitHub Issues
# =========================
def infer_ticket_category(title: str, body: str) -> str:
    text = f"{title} {body}".lower()

    if any(k in text for k in ["access", "permission", "repo", "organization"]):
        return "access_issue"
    if any(k in text for k in ["login", "account", "password", "sign in"]):
        return "account_issue"
    if any(k in text for k in ["network", "vpn", "connection"]):
        return "network_issue"
    if any(k in text for k in ["install", "setup", "tool", "software"]):
        return "software_issue"
    return "general_issue"


def infer_priority(title: str, body: str) -> str:
    text = f"{title} {body}".lower()

    if any(k in text for k in ["urgent", "critical", "blocked", "cannot access"]):
        return "high"
    if any(k in text for k in ["error", "fail", "issue", "problem"]):
        return "medium"
    return "low"


def build_issue_text(item: dict) -> str:
    title = (item.get("title") or "").strip()
    body = (item.get("body") or "").strip()
    state = (item.get("state") or "").strip()

    labels = []
    for label in item.get("labels", []) or []:
        if isinstance(label, dict):
            name = (label.get("name") or "").strip()
            if name:
                labels.append(name)

    parts = []
    if title:
        parts.append(f"Title: {title}")
    if state:
        parts.append(f"State: {state}")
    if labels:
        parts.append(f"Labels: {', '.join(labels)}")
    if body:
        parts.append(f"Body:\n{body}")

    return "\n\n".join(parts).strip()


def process_github_issues() -> list[dict]:
    issues_dir = RAW_DIR / "github_issues"
    tickets: list[dict] = []

    print(f"[DEBUG] issues_dir = {issues_dir}")
    print(f"[DEBUG] issues_dir exists = {issues_dir.exists()}")

    if not issues_dir.exists():
        print(f"[SKIP] Không tìm thấy {issues_dir}")
        return tickets

    merged_file = issues_dir / "github_docs_issues_all.json"
    page_files = sorted(issues_dir.glob("github_docs_issues_page*.json"))

    sources_to_read: list[Path] = []
    if merged_file.exists():
        sources_to_read = [merged_file]
    else:
        sources_to_read = page_files

    print(f"[DEBUG] using issue source files = {[p.name for p in sources_to_read]}")

    seen_issue_ids: set[str] = set()

    for json_file in sources_to_read:
        data = load_json_file(json_file)
        if not isinstance(data, list):
            continue

        issues_only = [item for item in data if isinstance(item, dict) and "pull_request" not in item]
        print(f"[DEBUG] issues_only from {json_file.name} = {len(issues_only)}")

        for item in issues_only:
            issue_id = str(item.get("id") or "").strip()
            if not issue_id:
                continue

            if issue_id in seen_issue_ids:
                continue
            seen_issue_ids.add(issue_id)

            title = (item.get("title") or "").strip()
            body = (item.get("body") or "").strip()
            text = build_issue_text(item)

            if len(text) < 30:
                continue

            labels = []
            for label in item.get("labels", []) or []:
                if isinstance(label, dict):
                    name = (label.get("name") or "").strip()
                    if name:
                        labels.append(name)

            tickets.append(
                {
                    "ticket_id": issue_id,
                    "issue_number": item.get("number"),
                    "source": "github_issues",
                    "source_type": "ticket",
                    "doc_type": "ticket",
                    "title": title,
                    "category": infer_ticket_category(title, body),
                    "priority": infer_priority(title, body),
                    "status": item.get("state", ""),
                    "labels": labels,
                    "created_at": item.get("created_at", ""),
                    "updated_at": item.get("updated_at", ""),
                    "url": item.get("html_url", ""),
                    "text": text,
                }
            )

    print(f"[DONE] GitHub tickets: {len(tickets)}")
    return tickets


# =========================
# Main
# =========================
def main():
    github_docs = process_github_docs()
    gitlab_docs = process_gitlab_handbook()
    github_tickets = process_github_issues()

    all_documents = github_docs + gitlab_docs

    save_jsonl(all_documents, PROCESSED_DIR / "documents.jsonl")
    save_jsonl(github_tickets, PROCESSED_DIR / "tickets.jsonl")

    stats = {
        "github_docs": len(github_docs),
        "gitlab_handbook": len(gitlab_docs),
        "github_issues": len(github_tickets),
        "total_documents": len(all_documents),
        "total_tickets": len(github_tickets),
    }
    save_json(stats, PROCESSED_DIR / "prepare_stats.json")

    print(f"[DONE] Saved {len(all_documents)} documents")
    print(f"[DONE] Saved {len(github_tickets)} tickets")
    print(f"[DONE] Saved stats -> {PROCESSED_DIR / 'prepare_stats.json'}")


if __name__ == "__main__":
    main()