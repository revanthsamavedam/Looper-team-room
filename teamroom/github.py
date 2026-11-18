"""Create a GitHub repo and push a project's files — plain REST API.

No MCP, no new dependencies: stdlib urllib against api.github.com.
The token comes from the GITHUB_TOKEN env var (a classic token with
the `repo` scope, or a fine-grained token with repository
Administration + Contents write). The token is never stored — it is
read from the environment for this one call only. When it is missing
the caller gets a clear setup error instead of a traceback.

Repo creation: POST /user/repos. Files: one PUT per file to
/repos/{owner}/{repo}/contents/{path} (a small project is a few
files; a large one would want the Git Data API instead, but the
contents API is the simple correct path here). Files are capped by
the store (MAX_PROJECT_FILE_CHARS), which keeps each file under the
contents API's 1 MB limit.
"""
from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.parse
import urllib.request

API_BASE = "https://api.github.com"
MISSING_TOKEN_MESSAGE = ("Add a repo-scoped GitHub token as GITHUB_TOKEN "
                         "in Vercel")


class GitHubError(Exception):
    """A clean, user-facing GitHub failure (no token, no traceback)."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def sanitize_repo_name(name: str) -> str:
    """Turn a project name into a valid GitHub repo name: lowercase,
    spaces/underscores to hyphens, anything else invalid dropped."""
    raw = (name or "").strip().lower()
    raw = re.sub(r"[\s_]+", "-", raw)
    raw = re.sub(r"[^a-z0-9.-]", "", raw)
    raw = raw.replace("..", "-")
    raw = re.sub(r"-+", "-", raw).strip("-.")
    return (raw[:100].strip("-.") or "teamroom-project")


def _request(token: str, method: str, path: str,
             payload: dict | None = None) -> dict:
    req = urllib.request.Request(API_BASE + path,
                                 data=(json.dumps(payload).encode()
                                       if payload is not None else None),
                                 method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if payload is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        try:
            message = json.loads(body).get("message", body) or body
        except Exception:  # noqa: BLE001 — fall back to the raw body
            message = body
        raise GitHubError(
            f"GitHub API {method} {path} failed ({exc.code}): {message}",
            status=exc.code)
    except urllib.error.URLError as exc:
        raise GitHubError(f"could not reach GitHub: {exc.reason}")


def create_repo_and_push(token: str | None, name: str, description: str,
                         files: list[dict], private: bool = True) -> dict:
    """Create a repo on the token owner's account and push the files.

    `files` is a list of {"path": ..., "content": ...}. Returns
    {"repo", "repo_url", "files_pushed", "private"}. Raises GitHubError
    with a plain message on any failure (missing token, bad token,
    name taken, API error). A failure part-way through pushing files
    reports how many files already landed.
    """
    token = (token or "").strip()
    if not token:
        raise GitHubError(MISSING_TOKEN_MESSAGE)
    repo_name = sanitize_repo_name(name)
    try:
        repo = _request(token, "POST", "/user/repos", {
            "name": repo_name,
            "description": (description or "").strip() or None,
            "private": bool(private),
            "auto_init": False,
        })
    except GitHubError as exc:
        if exc.status == 401:
            raise GitHubError(
                "GitHub token is invalid or expired (401) — check "
                "GITHUB_TOKEN in Vercel")
        if exc.status == 422:
            raise GitHubError(
                f"GitHub refused the repo name '{repo_name}' (422) — a repo "
                "with that name may already exist on your account")
        if exc.status == 403:
            raise GitHubError(
                "GitHub refused (403) — the token may lack the repo scope, "
                "or the API rate limit was hit")
        raise
    owner = (repo.get("owner") or {}).get("login") or ""
    full_name = repo.get("full_name") or f"{owner}/{repo_name}"
    pushed = 0
    for f in files:
        path = f["path"]
        encoded = urllib.parse.quote(path, safe="/")
        try:
            _request(token, "PUT",
                     f"/repos/{owner}/{repo_name}/contents/{encoded}", {
                         "message": f"Add {path}",
                         "content": base64.b64encode(
                             (f.get("content") or "").encode()).decode(),
                     })
        except GitHubError as exc:
            raise GitHubError(
                f"pushed {pushed} of {len(files)} files, then the push of "
                f"'{path}' failed: {exc}")
        pushed += 1
    return {"repo": full_name, "repo_url": repo.get("html_url") or "",
            "files_pushed": pushed, "private": bool(private)}
