"""Scan repo file tree and extract structured evidence."""

from pathlib import Path

import structlog

from flowedge.config.settings import get_settings
from flowedge.schemas.evidence import EvidenceItem, EvidencePack, EvidenceSource

logger = structlog.get_logger()

DOCS_DIRS = {"docs", "doc", "documentation", "wiki"}
EXAMPLE_DIRS = {"examples", "example", "samples", "notebooks"}
TEST_DIRS = {"tests", "test", "spec", "specs"}
DEP_FILES = {
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "Cargo.toml",
    "package.json",
    "go.mod",
}
README_NAMES = {"README.md", "README.rst", "README.txt", "README"}


def classify_file(path: Path, repo_root: Path) -> EvidenceSource | None:
    """Classify a file into an evidence source category."""
    rel = path.relative_to(repo_root)
    parts = rel.parts

    if rel.name in README_NAMES:
        return EvidenceSource.README
    if rel.name in DEP_FILES:
        return EvidenceSource.DEPS
    if any(p.lower() in DOCS_DIRS for p in parts):
        return EvidenceSource.DOCS
    if any(p.lower() in EXAMPLE_DIRS for p in parts):
        return EvidenceSource.EXAMPLES
    if any(p.lower() in TEST_DIRS for p in parts):
        return EvidenceSource.TESTS
    if path.suffix in {".py", ".rs", ".go", ".ts", ".js", ".java", ".cpp", ".c"}:
        return EvidenceSource.SOURCE
    return None


def scan_repo(repo_path: Path) -> EvidencePack:
    """Scan a cloned repo and build an evidence pack."""
    settings = get_settings()
    max_size = settings.max_file_size_kb * 1024
    repo_name = repo_path.name

    items: list[EvidenceItem] = []
    total_files = 0
    total_lines = 0
    languages: dict[str, int] = {}

    for file_path in repo_path.rglob("*"):
        if not file_path.is_file():
            continue
        if ".git" in file_path.parts:
            continue
        if file_path.stat().st_size > max_size:
            logger.debug("skipping_large_file", path=str(file_path))
            continue

        total_files += 1
        source = classify_file(file_path, repo_path)
        if source is None:
            continue

        ext = file_path.suffix or "other"
        languages[ext] = languages.get(ext, 0) + 1

        try:
            content = file_path.read_text(errors="replace")
            line_count = content.count("\n")
            total_lines += line_count
            snippet = content[:2000]

            items.append(
                EvidenceItem(
                    source=source,
                    file_path=str(file_path.relative_to(repo_path)),
                    content_snippet=snippet,
                    claim=f"File exists at {file_path.relative_to(repo_path)}",
                    confidence=1.0,
                )
            )
        except Exception:
            logger.warning("file_read_error", path=str(file_path))

    total_ext = sum(languages.values()) or 1
    lang_pct = {k: round(v / total_ext * 100, 1) for k, v in languages.items()}

    tree_lines = []
    for p in sorted(repo_path.iterdir()):
        prefix = "d " if p.is_dir() else "f "
        tree_lines.append(prefix + p.name)

    return EvidencePack(
        repo_url="",
        repo_name=repo_name,
        items=items,
        file_tree_summary="\n".join(tree_lines[:50]),
        language_breakdown=lang_pct,
        total_files=total_files,
        total_lines=total_lines,
    )
