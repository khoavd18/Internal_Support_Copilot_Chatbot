from __future__ import annotations

import os
from datetime import datetime

from src.core.settings import GITHUB_TEST_REPO
from src.integrations.github_client import GitHubClient


def main() -> int:
    repo = GITHUB_TEST_REPO
    if not repo:
        raise RuntimeError("Thiếu GITHUB_TEST_REPO trong .env")

    workflow_id = os.getenv("GITHUB_TEST_WORKFLOW_ID", "").strip()
    workflow_ref = os.getenv("GITHUB_TEST_WORKFLOW_REF", "main").strip()
    input_key = os.getenv("GITHUB_TEST_WORKFLOW_INPUT_KEY", "note").strip()
    input_value = os.getenv("GITHUB_TEST_WORKFLOW_INPUT_VALUE", "hello-from-copilot").strip()

    client = GitHubClient()

    print("=" * 80)
    print("1) Healthcheck")
    print(client.healthcheck(repo_full_name=repo))

    print("=" * 80)
    print("2) Create issue")
    issue_result = client.create_issue(
        repo_full_name=repo,
        title=f"Test issue from GitHub App - {datetime.utcnow().isoformat()}",
        body="Issue này được tạo bởi GitHub App integration test.",
        labels=[],
        assignees=[],
    )
    print(issue_result)

    issue_number = issue_result["issue_number"]

    print("=" * 80)
    print("3) Create issue comment")
    comment_result = client.create_issue_comment(
        repo_full_name=repo,
        issue_number=issue_number,
        body="Đây là comment test được tạo bởi GitHub App integration.",
    )
    print(comment_result)

    if workflow_id:
        print("=" * 80)
        print("4) Dispatch workflow")
        dispatch_result = client.dispatch_workflow(
            repo_full_name=repo,
            workflow_id=workflow_id,
            ref=workflow_ref,
            inputs={input_key: input_value},
        )
        print(dispatch_result)
    else:
        print("=" * 80)
        print("4) Skip dispatch workflow vì chưa có GITHUB_TEST_WORKFLOW_ID")

    print("=" * 80)
    print("SUCCESS: GitHubClient actions are working.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())