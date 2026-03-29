from pathlib import Path
import json
import os
import time
import hashlib
import subprocess
from collections import deque
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


# =========================
# PATHS
# =========================
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
BASE = PROJECT_ROOT / "data_source" / "raw"

GITHUB_DOCS_DIR = BASE / "github_docs"
GITHUB_DOCS_REPO_DIR = GITHUB_DOCS_DIR / "repo"
GITLAB_DIR = BASE / "gitlab_handbook"
GITLAB_HTML_DIR = GITLAB_DIR / "html"
GITHUB_ISSUES_DIR = BASE / "github_issues"

for folder in [GITHUB_DOCS_DIR, GITLAB_HTML_DIR, GITHUB_ISSUES_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# =========================
# CONFIG
# =========================
GITHUB_DOCS_REPO_URL = "https://github.com/github/docs.git"
GITHUB_DOCS_BRANCH = "main"

# content_only | full_repo
GITHUB_DOCS_MODE = os.getenv("GITHUB_DOCS_MODE", "content_only").strip()

# GitLab
GITLAB_MAX_PAGES = int(os.getenv("GITLAB_MAX_PAGES", "1000"))
GITLAB_SLEEP_SECONDS = float(os.getenv("GITLAB_SLEEP_SECONDS", "0.3"))

# GitHub Issues
GITHUB_ISSUES_PER_PAGE = 100
GITHUB_ISSUES_STATE = os.getenv("GITHUB_ISSUES_STATE", "all").strip()

REQUEST_TIMEOUT = 30

DEFAULT_HEADERS = {
    "User-Agent": "internal-support-copilot/1.0",
}

STATIC_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf",
    ".zip", ".mp4", ".webm", ".ico", ".css", ".js",
    ".woff", ".woff2", ".ttf", ".eot", ".xml",
}


# =========================
# HELPERS
# =========================
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path: Path, data) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_cmd(cmd, cwd=None, capture_output=False):
    print(f"[CMD] {' '.join(cmd)}")
    return subprocess.run(
        cmd,
        check=True,
        cwd=cwd,
        text=True,
        capture_output=capture_output,
    )


def ensure_git_longpaths():
    """
    Giảm lỗi path quá dài trên Windows.
    """
    try:
        run_cmd(["git", "config", "--global", "core.longpaths", "true"])
        print("[DONE] Enabled git core.longpaths=true")
    except Exception as e:
        print(f"[WARN] Could not set git core.longpaths=true: {e}")


def is_git_repo(path: Path) -> bool:
    return (path / ".git").exists()


def get_git_head_commit(repo_dir: Path) -> str | None:
    try:
        result = run_cmd(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def make_session(token: str | None = None) -> requests.Session:
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
    return session


def normalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    normalized = parsed._replace(fragment="", query="").geturl()
    if normalized.endswith("/"):
        normalized = normalized[:-1]
    return normalized


def safe_filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/")

    if not path:
        path = "index"

    path = path.replace("/", "__")
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    return f"{path}__{digest}.html"


def extract_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href:
            continue

        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)

        if parsed.scheme not in {"http", "https"}:
            continue

        links.append(normalize_url(full_url))

    return links


def should_skip_static_file(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in STATIC_EXTENSIONS)


# =========================
# GITHUB DOCS
# =========================
def init_sparse_repo(repo_dir: Path):
    repo_dir.mkdir(parents=True, exist_ok=True)

    run_cmd(["git", "init"], cwd=repo_dir)

    try:
        run_cmd(["git", "remote", "add", "origin", GITHUB_DOCS_REPO_URL], cwd=repo_dir)
    except Exception:
        # nếu remote đã tồn tại thì update lại URL
        run_cmd(["git", "remote", "set-url", "origin", GITHUB_DOCS_REPO_URL], cwd=repo_dir)

    run_cmd(["git", "sparse-checkout", "init", "--cone"], cwd=repo_dir)
    run_cmd(["git", "sparse-checkout", "set", "content"], cwd=repo_dir)


def sync_github_docs():
    """
    Đồng bộ github/docs.
    - content_only: sparse-checkout thư mục content
    - full_repo: shallow clone / fetch full repo
    """
    ensure_git_longpaths()

    if GITHUB_DOCS_MODE not in {"content_only", "full_repo"}:
        raise ValueError("GITHUB_DOCS_MODE phải là 'content_only' hoặc 'full_repo'")

    if GITHUB_DOCS_MODE == "full_repo":
        if not is_git_repo(GITHUB_DOCS_REPO_DIR):
            if GITHUB_DOCS_REPO_DIR.exists() and any(GITHUB_DOCS_REPO_DIR.iterdir()):
                raise RuntimeError(
                    f"Folder đã có dữ liệu nhưng không phải git repo: {GITHUB_DOCS_REPO_DIR}"
                )
            run_cmd([
                "git", "clone", "--depth", "1",
                "--branch", GITHUB_DOCS_BRANCH,
                GITHUB_DOCS_REPO_URL,
                str(GITHUB_DOCS_REPO_DIR),
            ])
            print("[DONE] Cloned FULL github/docs repo")
        else:
            run_cmd(["git", "fetch", "--depth", "1", "origin", GITHUB_DOCS_BRANCH], cwd=GITHUB_DOCS_REPO_DIR)
            run_cmd(["git", "checkout", GITHUB_DOCS_BRANCH], cwd=GITHUB_DOCS_REPO_DIR)
            run_cmd(["git", "reset", "--hard", f"origin/{GITHUB_DOCS_BRANCH}"], cwd=GITHUB_DOCS_REPO_DIR)
            print("[DONE] Updated FULL github/docs repo")
    else:
        if not is_git_repo(GITHUB_DOCS_REPO_DIR):
            init_sparse_repo(GITHUB_DOCS_REPO_DIR)

        run_cmd(["git", "fetch", "--depth", "1", "origin", GITHUB_DOCS_BRANCH], cwd=GITHUB_DOCS_REPO_DIR)
        try:
            run_cmd(["git", "checkout", GITHUB_DOCS_BRANCH], cwd=GITHUB_DOCS_REPO_DIR)
        except Exception:
            run_cmd(["git", "checkout", "-b", GITHUB_DOCS_BRANCH], cwd=GITHUB_DOCS_REPO_DIR)

        run_cmd(["git", "reset", "--hard", f"origin/{GITHUB_DOCS_BRANCH}"], cwd=GITHUB_DOCS_REPO_DIR)
        print("[DONE] Sparse checkout sync thư mục content thành công")

    metadata = {
        "repo_url": GITHUB_DOCS_REPO_URL,
        "branch": GITHUB_DOCS_BRANCH,
        "mode": GITHUB_DOCS_MODE,
        "updated_at": utc_now_iso(),
        "head_commit": get_git_head_commit(GITHUB_DOCS_REPO_DIR),
    }
    save_json(GITHUB_DOCS_DIR / "download_metadata.json", metadata)
    print(f"[DONE] Saved GitHub Docs metadata -> {GITHUB_DOCS_DIR / 'download_metadata.json'}")


# =========================
# GITLAB HANDBOOK
# =========================
def download_gitlab_handbook():
    """
    Crawl từ seed_urls.txt và đi tiếp theo internal links.
    Lưu:
    - html gốc
    - manifest giàu metadata
    """
    seed_file = GITLAB_DIR / "seed_urls.txt"
    manifest_path = GITLAB_DIR / "manifest.json"

    if not seed_file.exists():
        print(f"[SKIP] Không tìm thấy {seed_file}")
        return

    seed_urls = [
        normalize_url(x.strip())
        for x in seed_file.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    seed_urls = list(dict.fromkeys(seed_urls))

    if not seed_urls:
        print("[SKIP] seed_urls.txt rỗng")
        return

    manifest = load_json(
        manifest_path,
        {
            "last_run_started_at": None,
            "last_run_finished_at": None,
            "seed_urls": [],
            "visited_urls": [],
            "failed_urls": [],
            "pages": {},
            "errors": {},
        },
    )

    manifest["last_run_started_at"] = utc_now_iso()
    manifest["seed_urls"] = seed_urls

    visited = set(manifest.get("visited_urls", []))
    failed = set(manifest.get("failed_urls", []))
    pages = manifest.get("pages", {})
    errors = manifest.get("errors", {})

    allowed_netlocs = {urlparse(u).netloc for u in seed_urls}

    queue = deque()
    queued = set()

    for url in seed_urls:
        if url not in visited and url not in failed:
            queue.append(url)
            queued.add(url)

    downloaded_count = 0
    session = make_session()

    while queue and downloaded_count < GITLAB_MAX_PAGES:
        url = queue.popleft()
        queued.discard(url)

        if url in visited or url in failed:
            continue

        parsed = urlparse(url)
        if parsed.netloc not in allowed_netlocs:
            continue

        if should_skip_static_file(url):
            continue

        try:
            print(f"[FETCH] {url}")
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "").lower()
            if "text/html" not in content_type:
                print(f"[SKIP] Non-HTML content: {url} ({content_type})")
                failed.add(url)
                errors[url] = f"Non-HTML content: {content_type}"
                continue

            html = resp.text
            soup = BeautifulSoup(html, "html.parser")
            title = soup.title.get_text(" ", strip=True) if soup.title else ""

            file_name = safe_filename_from_url(url)
            out_file = GITLAB_HTML_DIR / file_name
            out_file.write_text(html, encoding="utf-8")

            visited.add(url)
            downloaded_count += 1

            pages[url] = {
                "file_name": file_name,
                "title": title,
                "fetched_at": utc_now_iso(),
                "status_code": resp.status_code,
                "content_type": content_type,
                "sha256": sha256_text(html),
            }

            print(f"[DONE] Saved {url} -> {file_name}")

            for link in extract_links(html, url):
                link_parsed = urlparse(link)

                if link_parsed.netloc not in allowed_netlocs:
                    continue

                if should_skip_static_file(link):
                    continue

                if link not in visited and link not in failed and link not in queued:
                    queue.append(link)
                    queued.add(link)

            manifest["visited_urls"] = sorted(visited)
            manifest["failed_urls"] = sorted(failed)
            manifest["pages"] = pages
            manifest["errors"] = errors
            save_json(manifest_path, manifest)

            time.sleep(GITLAB_SLEEP_SECONDS)

        except Exception as e:
            print(f"[ERROR] Failed to download {url}: {e}")
            failed.add(url)
            errors[url] = str(e)

            manifest["visited_urls"] = sorted(visited)
            manifest["failed_urls"] = sorted(failed)
            manifest["pages"] = pages
            manifest["errors"] = errors
            save_json(manifest_path, manifest)

    manifest["last_run_finished_at"] = utc_now_iso()
    manifest["visited_urls"] = sorted(visited)
    manifest["failed_urls"] = sorted(failed)
    manifest["pages"] = pages
    manifest["errors"] = errors
    save_json(manifest_path, manifest)

    print(f"[DONE] GitLab Handbook crawl finished. Downloaded pages this run: {downloaded_count}")
    print(f"[INFO] Total visited URLs: {len(visited)}")
    print(f"[INFO] Total failed URLs: {len(failed)}")


# =========================
# GITHUB ISSUES
# =========================
def download_github_issues():
    """
    Download toàn bộ issues qua pagination.
    - Có hỗ trợ GITHUB_TOKEN
    - Mỗi run sẽ overwrite raw issue pages để tránh lẫn dữ liệu cũ
    """
    token = os.getenv("GITHUB_TOKEN", "").strip()
    session = make_session(token)

    # dọn page cũ để dữ liệu luôn đồng nhất theo mỗi lần chạy
    for old_file in GITHUB_ISSUES_DIR.glob("github_docs_issues_page*.json"):
        old_file.unlink(missing_ok=True)

    all_issues = []
    page = 1
    saved_pages = 0

    while True:
        issues_url = (
            "https://api.github.com/repos/github/docs/issues"
            f"?state={GITHUB_ISSUES_STATE}"
            f"&per_page={GITHUB_ISSUES_PER_PAGE}"
            f"&page={page}"
        )

        try:
            print(f"[FETCH] GitHub issues page {page}")
            resp = session.get(
                issues_url,
                timeout=REQUEST_TIMEOUT,
                headers={
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )
            resp.raise_for_status()

            data = resp.json()
            if not data:
                print("[DONE] No more issue pages.")
                break

            page_file = GITHUB_ISSUES_DIR / f"github_docs_issues_page{page}.json"
            save_json(page_file, data)
            print(f"[DONE] Saved page {page} -> {page_file.name}")

            only_issues = [item for item in data if "pull_request" not in item]
            all_issues.extend(only_issues)

            saved_pages += 1
            page += 1
            time.sleep(0.2)

        except Exception as e:
            print(f"[ERROR] Failed to download GitHub issues page {page}: {e}")
            break

    merged_file = GITHUB_ISSUES_DIR / "github_docs_issues_all.json"
    save_json(merged_file, all_issues)

    metadata = {
        "repo": "github/docs",
        "state": GITHUB_ISSUES_STATE,
        "per_page": GITHUB_ISSUES_PER_PAGE,
        "saved_pages": saved_pages,
        "total_issues_without_prs": len(all_issues),
        "updated_at": utc_now_iso(),
    }
    save_json(GITHUB_ISSUES_DIR / "download_metadata.json", metadata)

    print(f"[DONE] Saved merged issues -> {merged_file}")
    print(f"[INFO] Total saved pages: {saved_pages}")
    print(f"[INFO] Total issues (without PRs): {len(all_issues)}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    sync_github_docs()
    download_gitlab_handbook()
    download_github_issues()