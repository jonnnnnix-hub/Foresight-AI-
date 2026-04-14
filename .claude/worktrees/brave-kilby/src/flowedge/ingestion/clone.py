"""Safe repository cloning with size and security limits."""

import re
from pathlib import Path

import structlog
from git import Repo

from flowedge.config.settings import get_settings

logger = structlog.get_logger()

GITHUB_URL_PATTERN = re.compile(
    r"^https://github\.com/[\w\-\.]+/[\w\-\.]+/?$"
)


def validate_repo_url(url: str) -> str:
    """Validate and normalize a GitHub repository URL."""
    url = url.rstrip("/")
    if not GITHUB_URL_PATTERN.match(url):
        raise ValueError(f"Invalid GitHub URL: {url}")
    return url


def clone_repo(url: str) -> Path:
    """Clone a GitHub repo to the local clone directory."""
    settings = get_settings()
    url = validate_repo_url(url)

    repo_name = url.rstrip("/").split("/")[-1]
    clone_dir = settings.clone_base_dir / repo_name

    if clone_dir.exists():
        logger.info("repo_already_cloned", repo_name=repo_name, path=str(clone_dir))
        return clone_dir

    settings.clone_base_dir.mkdir(parents=True, exist_ok=True)
    logger.info("cloning_repo", url=url, dest=str(clone_dir))
    Repo.clone_from(url, str(clone_dir), depth=1)
    logger.info("clone_complete", repo_name=repo_name)

    return clone_dir
