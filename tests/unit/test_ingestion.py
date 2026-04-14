"""Tests for ingestion validation."""

import pytest

from flowedge.ingestion.clone import validate_repo_url


def test_valid_github_url() -> None:
    url = validate_repo_url("https://github.com/user/repo")
    assert url == "https://github.com/user/repo"


def test_valid_github_url_trailing_slash() -> None:
    url = validate_repo_url("https://github.com/user/repo/")
    assert url == "https://github.com/user/repo"


def test_invalid_url_raises() -> None:
    with pytest.raises(ValueError, match="Invalid GitHub URL"):
        validate_repo_url("https://gitlab.com/user/repo")


def test_invalid_format_raises() -> None:
    with pytest.raises(ValueError, match="Invalid GitHub URL"):
        validate_repo_url("not-a-url")
