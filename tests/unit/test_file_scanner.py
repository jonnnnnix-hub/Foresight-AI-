"""Tests for file scanner and evidence extraction."""

from pathlib import Path

from flowedge.extraction.file_scanner import classify_file, scan_repo
from flowedge.schemas.evidence import EvidenceSource

FIXTURE_REPO = Path(__file__).parent.parent / "fixtures" / "fake_repo"


def test_classify_readme() -> None:
    path = FIXTURE_REPO / "README.md"
    assert classify_file(path, FIXTURE_REPO) == EvidenceSource.README


def test_classify_source() -> None:
    path = FIXTURE_REPO / "src" / "main.py"
    assert classify_file(path, FIXTURE_REPO) == EvidenceSource.SOURCE


def test_classify_tests() -> None:
    path = FIXTURE_REPO / "tests" / "test_basic.py"
    assert classify_file(path, FIXTURE_REPO) == EvidenceSource.TESTS


def test_classify_docs() -> None:
    path = FIXTURE_REPO / "docs" / "guide.md"
    assert classify_file(path, FIXTURE_REPO) == EvidenceSource.DOCS


def test_classify_examples() -> None:
    path = FIXTURE_REPO / "examples" / "demo.py"
    assert classify_file(path, FIXTURE_REPO) == EvidenceSource.EXAMPLES


def test_classify_deps() -> None:
    path = FIXTURE_REPO / "pyproject.toml"
    assert classify_file(path, FIXTURE_REPO) == EvidenceSource.DEPS


def test_classify_unknown_returns_none() -> None:
    path = FIXTURE_REPO / "random.xyz"
    assert classify_file(path, FIXTURE_REPO) is None


def test_scan_repo_produces_evidence_pack() -> None:
    pack = scan_repo(FIXTURE_REPO)
    assert pack.repo_name == "fake_repo"
    assert pack.total_files > 0
    assert len(pack.items) > 0
    # Should find at least README, source, tests, docs, examples, deps
    sources_found = {item.source for item in pack.items}
    assert EvidenceSource.README in sources_found
    assert EvidenceSource.SOURCE in sources_found
    assert EvidenceSource.TESTS in sources_found


def test_scan_repo_language_breakdown() -> None:
    pack = scan_repo(FIXTURE_REPO)
    assert len(pack.language_breakdown) > 0
    assert ".py" in pack.language_breakdown


def test_scan_repo_file_tree() -> None:
    pack = scan_repo(FIXTURE_REPO)
    assert pack.file_tree_summary != ""
    assert "src" in pack.file_tree_summary or "d src" in pack.file_tree_summary
