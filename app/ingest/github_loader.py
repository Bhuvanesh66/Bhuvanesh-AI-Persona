"""Pull the persona's GitHub corpus via the REST API.

For each repo we ingest:
  - README                (chunk_type='readme')
  - a bounded set of source files (chunk_type='code')
  - a condensed commit log (chunk_type='commit') — for "commit-only" eval questions
  - a metadata blurb       (chunk_type='meta')
Forks are tagged is_fork=True so the persona never claims authorship (see PRD §5.1.1).
"""
from __future__ import annotations

import base64
from dataclasses import dataclass

import requests

from app.config import settings

API = "https://api.github.com"

CODE_EXTS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".rb",
    ".c", ".cpp", ".cs", ".sql", ".sh", ".html", ".css", ".md", ".yaml", ".yml",
}
SKIP_SUBSTR = (
    "node_modules/", "dist/", "build/", ".min.", "package-lock.json",
    "yarn.lock", "pnpm-lock.yaml", "vendor/", ".lock", "/.", "test/", "tests/",
)
MAX_FILE_BYTES = 100_000
MAX_CODE_FILES = 10
MAX_COMMITS = 80


@dataclass
class Doc:
    source: str
    repo: str | None
    file_path: str | None
    chunk_type: str
    is_fork: bool
    title: str
    text: str


def _headers() -> dict:
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if settings.github_token:
        h["Authorization"] = f"Bearer {settings.github_token}"
    return h


def _get(url: str, **params):
    r = requests.get(url, headers=_headers(), params=params, timeout=30)
    if r.status_code == 404:
        return None
    if r.status_code == 403 and "rate limit" in r.text.lower():
        raise RuntimeError("GitHub rate limit hit. Set GITHUB_TOKEN in .env (60→5000 req/hr).")
    r.raise_for_status()
    return r.json()


def list_repos() -> list[dict]:
    repos, page = [], 1
    while True:
        batch = _get(f"{API}/users/{settings.github_owner}/repos",
                     per_page=100, page=page, sort="updated")
        if not batch:
            break
        repos.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return repos


def _readme(owner: str, repo: str) -> str | None:
    data = _get(f"{API}/repos/{owner}/{repo}/readme")
    if not data or data.get("encoding") != "base64":
        return None
    try:
        return base64.b64decode(data["content"]).decode("utf-8", "replace")
    except Exception:
        return None


def _tree_files(owner: str, repo: str, branch: str) -> list[dict]:
    data = _get(f"{API}/repos/{owner}/{repo}/git/trees/{branch}", recursive=1)
    if not data or "tree" not in data:
        return []
    files = []
    for node in data["tree"]:
        if node.get("type") != "blob":
            continue
        path = node["path"]
        lower = path.lower()
        if any(s in lower for s in SKIP_SUBSTR):
            continue
        if not any(lower.endswith(ext) for ext in CODE_EXTS):
            continue
        if lower.endswith("readme.md"):
            continue  # handled separately
        if node.get("size", 0) > MAX_FILE_BYTES:
            continue
        files.append(node)
    # Prefer larger, likely-substantive files; cap the count.
    files.sort(key=lambda n: n.get("size", 0), reverse=True)
    return files[:MAX_CODE_FILES]


def _file_content(owner: str, repo: str, path: str, ref: str) -> str | None:
    data = _get(f"{API}/repos/{owner}/{repo}/contents/{path}", ref=ref)
    if not data or data.get("encoding") != "base64":
        return None
    try:
        return base64.b64decode(data["content"]).decode("utf-8", "replace")
    except Exception:
        return None


def _commit_log(owner: str, repo: str) -> str | None:
    lines, page = [], 1
    while len(lines) < MAX_COMMITS:
        batch = _get(f"{API}/repos/{owner}/{repo}/commits", per_page=100, page=page)
        if not batch:
            break
        for c in batch:
            sha = c.get("sha", "")[:7]
            msg = (c.get("commit", {}).get("message") or "").splitlines()
            subject = msg[0].strip() if msg else ""
            if subject:
                lines.append(f"- {sha}: {subject}")
            if len(lines) >= MAX_COMMITS:
                break
        if len(batch) < 100:
            break
        page += 1
    return "\n".join(lines) if lines else None


def load_repo(repo: dict) -> list[Doc]:
    owner = settings.github_owner
    name = repo["name"]
    is_fork = bool(repo.get("fork"))
    branch = repo.get("default_branch") or "main"
    docs: list[Doc] = []

    kind = "FORK (open-source project — NOT authored from scratch)" if is_fork else "original project"
    meta = (
        f"Repository: {name}\n"
        f"Type: {kind}\n"
        f"Language: {repo.get('language') or 'n/a'}\n"
        f"Description: {repo.get('description') or 'n/a'}\n"
        f"URL: {repo.get('html_url')}\n"
        f"Stars: {repo.get('stargazers_count', 0)} | Forks: {repo.get('forks_count', 0)}"
    )
    docs.append(Doc("github", name, "meta", "meta", is_fork, f"{name} — metadata", meta))

    readme = _readme(owner, name)
    if readme:
        docs.append(Doc("github", name, "README.md", "readme", is_fork, f"{name} — README", readme))

    for node in _tree_files(owner, name, branch):
        content = _file_content(owner, name, node["path"], branch)
        if content and content.strip():
            docs.append(Doc("github", name, node["path"], "code", is_fork,
                            f"{name} — {node['path']}", content))

    log = _commit_log(owner, name)
    if log:
        docs.append(Doc("github", name, "commit-log", "commit", is_fork,
                        f"{name} — commit history", log))

    return docs


def load_all() -> list[Doc]:
    docs: list[Doc] = []
    repos = list_repos()
    for i, repo in enumerate(repos, 1):
        tag = "fork" if repo.get("fork") else "orig"
        print(f"  [{i}/{len(repos)}] {repo['name']} ({tag})")
        try:
            docs.extend(load_repo(repo))
        except Exception as e:  # one bad repo shouldn't kill the run
            print(f"      ! skipped {repo['name']}: {e}")
    return docs
